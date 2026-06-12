#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study — Configurable CARCM
====================================
Self-contained copy of the relevant architecture from rice_classifier_v7b.py,
extended with `AblationCARCM`, a single configurable model that supports:

  * removing the context branch entirely (sequence-only)
  * keeping only one context dimension (variety / tissue / stage)
  * disabling interaction features (variety_tissue, tissue_stage, variety_stage)
  * removing the sequence branch (context-only baseline)
  * replacing the context branch with a learnable bias vector of equivalent
    parameter count (parameter-matched control)

This module does NOT modify any code outside the Ablation study/ directory.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


MOTIF_NAMES = [
    "ABRE", "ABRE_core", "CE1", "CE3", "DRE_CRT", "DRE_full", "LTRE",
    "MYB_core", "MYB1AT", "MYB2", "MYBCORE", "MYC", "WBOX", "WBOX_core",
    "GT1", "GT1_core", "HSE", "HSE_core", "GCC_box", "ERE", "GBOX", "IBOX",
    "CAAT_box", "STRE", "LTRE_like", "JASE", "TBOX", "NAC_core", "NACRS",
    "ACGT_core", "ABRE_like", "AuxRE", "AuxRE_rev", "ROOTMOTIF", "AS1",
    "TATA_box", "CACTFTPPCA1", "ICE_box", "CBF_DRE", "ACGT", "CACG", "CATG",
    "CGTG", "GCGT", "AACA", "CAAC", "AAAG", "GATA", "TATA", "CAAT", "GCCG",
    "CGCG", "TGAC", "GTAC",
]
NUM_BINS = 5


# =============================================================================
# DATA HELPERS  (verbatim from rice_classifier_v7b.py except for header)
# =============================================================================

def load_feature_groups(data_dir):
    with open(os.path.join(data_dir, "feature_groups.json"), "r") as f:
        return json.load(f)


def create_context_interactions(df, variety_cols, tissue_cols, stage_cols, vocab=None):
    """
    Build the variety×tissue, tissue×stage, variety×stage interaction one-hots
    for `df`. If `vocab` is None, vocabularies are *fitted* on this dataframe
    and returned alongside the one-hots; otherwise the supplied `vocab` is
    used to *transform* this dataframe so the resulting one-hots are
    column-compatible with the training pool. Indices that fall outside the
    fitted vocabulary collapse to all-zero rows.
    """
    variety_idx = df[variety_cols].values.argmax(axis=1)
    tissue_idx = df[tissue_cols].values.argmax(axis=1)
    stage_idx = df[stage_cols].values.argmax(axis=1)

    n_tissues = len(tissue_cols)
    n_stages = len(stage_cols)

    vt_idx = variety_idx * n_tissues + tissue_idx
    ts_idx = tissue_idx * n_stages + stage_idx
    vs_idx = variety_idx * n_stages + stage_idx

    if vocab is None:
        vt_unique = np.unique(vt_idx)
        ts_unique = np.unique(ts_idx)
        vs_unique = np.unique(vs_idx)
        vt_map = {int(v): i for i, v in enumerate(vt_unique)}
        ts_map = {int(v): i for i, v in enumerate(ts_unique)}
        vs_map = {int(v): i for i, v in enumerate(vs_unique)}
        vocab = {
            "vt_map": vt_map, "n_vt": len(vt_unique),
            "ts_map": ts_map, "n_ts": len(ts_unique),
            "vs_map": vs_map, "n_vs": len(vs_unique),
        }
    vt_map, ts_map, vs_map = vocab["vt_map"], vocab["ts_map"], vocab["vs_map"]
    n_vt, n_ts, n_vs = vocab["n_vt"], vocab["n_ts"], vocab["n_vs"]

    def _onehot(idx_arr, mapping, width):
        out = np.zeros((len(idx_arr), width), dtype=np.float32)
        for r, raw in enumerate(idx_arr):
            mapped = mapping.get(int(raw), -1)
            if mapped >= 0:
                out[r, mapped] = 1.0
        return out

    vt_onehot = _onehot(vt_idx, vt_map, n_vt)
    ts_onehot = _onehot(ts_idx, ts_map, n_ts)
    vs_onehot = _onehot(vs_idx, vs_map, n_vs)

    return {
        "variety_tissue": vt_onehot, "n_vt": n_vt,
        "tissue_stage": ts_onehot, "n_ts": n_ts,
        "variety_stage": vs_onehot, "n_vs": n_vs,
        "variety_idx": variety_idx,
        "tissue_idx": tissue_idx,
        "stage_idx": stage_idx,
        "vocab": vocab,
    }


def identify_hard_examples(train_df, test_df, count_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_counts = np.round(train_df[count_cols].values, 4)
    train_hashes = [hash(tuple(row)) for row in train_counts]
    train_df["motif_hash"] = train_hashes
    conflicting = train_df.groupby("motif_hash")["label"].nunique()
    hard_promoters = set(conflicting[conflicting > 1].index)
    train_hard = train_df["motif_hash"].isin(hard_promoters).values
    test_counts = np.round(test_df[count_cols].values, 4)
    test_hashes = [hash(tuple(row)) for row in test_counts]
    test_df["motif_hash"] = test_hashes
    test_hard = test_df["motif_hash"].isin(hard_promoters).values
    return train_hard, test_hard, hard_promoters


def permute_context_columns(df, variety_cols, tissue_cols, stage_cols, seed):
    """
    Return a copy of df with each context dimension's one-hot block independently
    permuted across rows. Variety, tissue, and stage are shuffled independently.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for cols in (variety_cols, tissue_cols, stage_cols):
        perm = rng.permutation(len(out))
        out.loc[:, cols] = out[cols].values[perm]
    return out


class PositionalMotifDataset(Dataset):
    def __init__(self, df, feature_groups, variety_cols, tissue_cols, stage_cols,
                 context_info=None, hard_examples=None,
                 fit_scaler=False, scaler=None):

        binned_cols = feature_groups["binned"]
        relpos_cols = feature_groups["relative_position"]
        firstpos_cols = feature_groups["first_position"]
        lastpos_cols = feature_groups["last_position"]
        gradient_cols = feature_groups["density_gradient"]
        count_cols = feature_groups["global_count"]
        cooc_cols = feature_groups["cooccurrence"]
        seq_stat_cols = feature_groups["seq_stats"]

        self.binned_data = df[binned_cols].values.astype(np.float32)
        self.relpos_data = df[relpos_cols].values.astype(np.float32)
        self.firstpos_data = df[firstpos_cols].values.astype(np.float32)
        self.lastpos_data = df[lastpos_cols].values.astype(np.float32)
        self.gradient_data = df[gradient_cols].values.astype(np.float32)
        self.count_data = df[count_cols].values.astype(np.float32)
        self.cooc_data = df[cooc_cols].values.astype(np.float32)
        self.seq_stats = df[seq_stat_cols].values.astype(np.float32)

        self.variety_data = df[variety_cols].values.astype(np.float32)
        self.tissue_data = df[tissue_cols].values.astype(np.float32)
        self.stage_data = df[stage_cols].values.astype(np.float32)
        self.labels = df["label"].values.astype(np.float32)

        if context_info is not None:
            self.variety_tissue = context_info["variety_tissue"]
            self.tissue_stage = context_info["tissue_stage"]
            self.variety_stage = context_info["variety_stage"]
            self.variety_idx = context_info["variety_idx"]
            self.tissue_idx = context_info["tissue_idx"]
            self.stage_idx = context_info["stage_idx"]
        else:
            self.variety_tissue = None

        if hard_examples is not None:
            self.is_hard = hard_examples.astype(np.float32)
        else:
            self.is_hard = np.zeros(len(self.labels), dtype=np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            scalable = np.hstack([self.binned_data, self.count_data,
                                  self.cooc_data, self.seq_stats])
            scalable_scaled = self.scaler.fit_transform(scalable)
        elif scaler is not None:
            self.scaler = scaler
            scalable = np.hstack([self.binned_data, self.count_data,
                                  self.cooc_data, self.seq_stats])
            scalable_scaled = self.scaler.transform(scalable)
        else:
            self.scaler = None
            scalable_scaled = None

        if scalable_scaled is not None:
            n_binned = self.binned_data.shape[1]
            n_count = self.count_data.shape[1]
            n_cooc = self.cooc_data.shape[1]
            self.binned_data = scalable_scaled[:, :n_binned].astype(np.float32)
            self.count_data = scalable_scaled[:, n_binned:n_binned + n_count].astype(np.float32)
            self.cooc_data = scalable_scaled[:, n_binned + n_count:n_binned + n_count + n_cooc].astype(np.float32)
            self.seq_stats = scalable_scaled[:, n_binned + n_count + n_cooc:].astype(np.float32)

        self.n_binned = self.binned_data.shape[1]
        self.n_relpos = self.relpos_data.shape[1]
        self.n_cooc = self.cooc_data.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "binned": torch.FloatTensor(self.binned_data[idx]),
            "relpos": torch.FloatTensor(self.relpos_data[idx]),
            "firstpos": torch.FloatTensor(self.firstpos_data[idx]),
            "lastpos": torch.FloatTensor(self.lastpos_data[idx]),
            "gradient": torch.FloatTensor(self.gradient_data[idx]),
            "count": torch.FloatTensor(self.count_data[idx]),
            "cooc": torch.FloatTensor(self.cooc_data[idx]),
            "seq_stats": torch.FloatTensor(self.seq_stats[idx]),
            "variety": torch.FloatTensor(self.variety_data[idx]),
            "tissue": torch.FloatTensor(self.tissue_data[idx]),
            "stage": torch.FloatTensor(self.stage_data[idx]),
            "label": torch.FloatTensor([self.labels[idx]]),
            "is_hard": torch.FloatTensor([self.is_hard[idx]]),
        }
        if self.variety_tissue is not None:
            item["variety_tissue"] = torch.FloatTensor(self.variety_tissue[idx])
            item["tissue_stage"] = torch.FloatTensor(self.tissue_stage[idx])
            item["variety_stage"] = torch.FloatTensor(self.variety_stage[idx])
            item["variety_idx"] = torch.LongTensor([self.variety_idx[idx]])
            item["tissue_idx"] = torch.LongTensor([self.tissue_idx[idx]])
            item["stage_idx"] = torch.LongTensor([self.stage_idx[idx]])
        return item


# =============================================================================
# ARCHITECTURE COMPONENTS  (copied from rice_classifier_v7b.py)
# =============================================================================

class PositionalMotifProcessor(nn.Module):
    """
    Combines two motif feature paths inside a single hidden_dim output:
      (a) Binned Motif Counts: bin_attention + bin_conv on (n_motifs, n_bins)
      (b) Positional Features: relpos / firstpos / lastpos / gradient via pos_fc

    `use_binned` / `use_positional` allow either sub-path to be removed
    architecturally for the feature-level ablation. At least one must be True.
    """

    def __init__(self, n_motifs, n_bins, hidden_dim, dropout=0.4,
                 use_binned=True, use_positional=True):
        super().__init__()
        assert use_binned or use_positional, \
            "PositionalMotifProcessor requires at least one of binned/positional"
        self.n_motifs = n_motifs
        self.n_bins = n_bins
        self.use_binned = use_binned
        self.use_positional = use_positional

        if use_binned:
            self.bin_conv = nn.Sequential(
                nn.Conv1d(n_motifs, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.bin_attention = nn.Sequential(
                nn.Linear(n_bins, n_bins),
                nn.Softmax(dim=-1),
            )

        if use_positional:
            self.pos_fc = nn.Sequential(
                nn.Linear(n_motifs * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        in_dim = hidden_dim * (int(use_binned) + int(use_positional))
        self.combine = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, binned, relpos, firstpos, lastpos, gradient):
        parts = []
        if self.use_binned:
            batch_size = binned.shape[0]
            binned_3d = binned.view(batch_size, self.n_motifs, self.n_bins)
            bin_weights = self.bin_attention(binned_3d)
            binned_weighted = binned_3d * bin_weights
            bin_features = self.bin_conv(binned_weighted)
            bin_features = bin_features.mean(dim=2)
            parts.append(bin_features)
        if self.use_positional:
            pos_concat = torch.cat([relpos, firstpos, lastpos, gradient], dim=1)
            pos_features = self.pos_fc(pos_concat)
            parts.append(pos_features)
        combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.combine(combined)


# =============================================================================
# CONFIGURABLE ABLATION MODEL
# =============================================================================

class AblationCARCM(nn.Module):
    """
    Configurable variant of ContextEnhancedGatedFusionV5 supporting all 8
    ablation rows.

    Switches
    --------
    use_sequence : bool
        If False, the entire sequence branch (positional motif + cooccurrence
        + count-LSTM) is removed. The output head consumes only context_feat.
    context_dims : list[str]
        Subset of {"variety", "tissue", "stage"} to include. Empty list removes
        the context branch entirely (configuration #2: sequence-only).
    use_interactions : bool
        Whether to include the variety_tissue / tissue_stage / variety_stage
        interaction one-hots. Only relevant if 2+ context dimensions are kept.
    bias_replace : bool
        If True, the entire context branch is replaced by a parameter-matched
        learnable bias vector that is broadcast across the batch. The bias's
        parameter count is matched to the original (full) context branch.
    """

    def __init__(self, n_motifs, n_bins, n_cooc, n_varieties, n_tissues, n_stages,
                 num_seq_stats=4, hidden_dim=128, embed_dim=64, dropout=0.4,
                 n_vt=0, n_ts=0, n_vs=0,
                 use_sequence=True, context_dims=("variety", "tissue", "stage"),
                 use_interactions=True, bias_replace=False,
                 feature_mask=None):
        super().__init__()

        self.n_motifs = n_motifs
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.use_sequence = use_sequence
        self.context_dims = list(context_dims)
        self.use_interactions = use_interactions and len(self.context_dims) >= 2
        self.bias_replace = bias_replace

        # feature_mask gates the four sequence-side feature blocks
        # (only meaningful when use_sequence=True). Default = all on.
        default_mask = {"binned": True, "positional": True,
                        "count": True, "cooccurrence": True}
        if feature_mask is None:
            feature_mask = default_mask
        else:
            for k, v in default_mask.items():
                feature_mask.setdefault(k, v)
        self.feature_mask = feature_mask
        self._use_binned = bool(feature_mask["binned"])
        self._use_positional = bool(feature_mask["positional"])
        self._use_count = bool(feature_mask["count"])
        self._use_cooc = bool(feature_mask["cooccurrence"])

        # ---------------- sequence branch ----------------
        if self.use_sequence:
            assert (self._use_binned or self._use_positional or
                    self._use_count or self._use_cooc), \
                "use_sequence=True but feature_mask disables every sequence path"

            if self._use_binned or self._use_positional:
                self.positional_proc = PositionalMotifProcessor(
                    n_motifs, n_bins, hidden_dim, dropout,
                    use_binned=self._use_binned,
                    use_positional=self._use_positional,
                )

            if self._use_cooc:
                self.cooc_fc = nn.Sequential(
                    nn.Linear(n_cooc, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                )

            if self._use_count:
                self.count_embed = nn.Linear(1, 32)
                self.lstm = nn.LSTM(32, hidden_dim // 2, num_layers=2,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=True)
                self.count_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # ---------------- context branch ----------------
        has_context = (len(self.context_dims) > 0) or self.bias_replace

        if self.bias_replace:
            # Replace the entire context branch with a learnable bias-only module
            # that produces a constant `context_feat` of shape (B, hidden_dim).
            # Parameter count is matched to the full context branch (computed below).
            full_param_count = self._estimate_full_context_params(
                n_varieties, n_tissues, n_stages,
                n_vt=n_vt, n_ts=n_ts, n_vs=n_vs,
                num_seq_stats=num_seq_stats, hidden_dim=hidden_dim, embed_dim=embed_dim
            )
            # Allocate a parameter tensor of size full_param_count.
            # The first hidden_dim entries are used as the bias vector;
            # the remainder is held as redundant trainable parameters to match
            # the parameter budget of the original context branch.
            self.bias_vec = nn.Parameter(torch.randn(hidden_dim) * 0.1)
            extra = max(0, full_param_count - hidden_dim)
            self.bias_pad = nn.Parameter(torch.zeros(extra))
            # context_query / motif_key / motif_value still used downstream
            self.context_query = nn.Linear(hidden_dim, hidden_dim)

        elif has_context:
            # Build the standard embedding + interaction context branch,
            # but only over the context dims that are enabled.
            self.use_variety = "variety" in self.context_dims
            self.use_tissue = "tissue" in self.context_dims
            self.use_stage = "stage" in self.context_dims

            if self.use_variety:
                self.variety_embed = nn.Embedding(n_varieties, embed_dim)
                nn.init.normal_(self.variety_embed.weight, 0, 0.1)
            if self.use_tissue:
                self.tissue_embed = nn.Embedding(n_tissues, embed_dim)
                nn.init.normal_(self.tissue_embed.weight, 0, 0.1)
            if self.use_stage:
                self.stage_embed = nn.Embedding(n_stages, embed_dim)
                nn.init.normal_(self.stage_embed.weight, 0, 0.1)

            embed_total = embed_dim * sum([self.use_variety, self.use_tissue, self.use_stage])
            inter_total = 0
            if self.use_interactions:
                if self.use_variety and self.use_tissue:
                    inter_total += n_vt
                if self.use_tissue and self.use_stage:
                    inter_total += n_ts
                if self.use_variety and self.use_stage:
                    inter_total += n_vs

            context_input_dim = embed_total + inter_total + num_seq_stats
            self.context_fc = nn.Sequential(
                nn.Linear(context_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.context_query = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Sequence-only: no context branch at all.
            self.use_variety = False
            self.use_tissue = False
            self.use_stage = False

        # ---------------- attention/gating + head ----------------
        if self.use_sequence:
            # motif_combined dim depends on which sequence paths are active.
            # pos_feat: hidden_dim (if binned or positional)
            # cooc_feat: hidden_dim // 2 (if cooccurrence)
            # count_feat: hidden_dim // 2 (if count)
            motif_feature_dim = (
                (hidden_dim if (self._use_binned or self._use_positional) else 0)
                + (hidden_dim // 2 if self._use_cooc else 0)
                + (hidden_dim // 2 if self._use_count else 0)
            )
            assert motif_feature_dim > 0, "all sequence feature paths are off"
            self._motif_feature_dim = motif_feature_dim
            self.motif_key = nn.Linear(motif_feature_dim, hidden_dim)
            self.motif_value = nn.Linear(motif_feature_dim, hidden_dim)
            self.attn_scale = hidden_dim ** 0.5

        if self.use_sequence and has_context:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self._has_context = has_context

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_full_context_params(n_varieties, n_tissues, n_stages,
                                      n_vt, n_ts, n_vs, num_seq_stats,
                                      hidden_dim, embed_dim):
        """Approximate parameter count of the full (variety+tissue+stage+inter)
        context branch including embeddings, context_fc and context_query."""
        emb = (n_varieties + n_tissues + n_stages) * embed_dim
        ctx_in = embed_dim * 3 + n_vt + n_ts + n_vs + num_seq_stats
        fc1 = ctx_in * hidden_dim + hidden_dim       # Linear
        ln = 2 * hidden_dim                          # LayerNorm
        fc2 = hidden_dim * hidden_dim + hidden_dim   # Linear
        cq = hidden_dim * hidden_dim + hidden_dim    # context_query Linear
        return emb + fc1 + ln + fc2 + cq

    # ------------------------------------------------------------------
    def _build_context_feat(self, batch, B, device):
        """Return context_feat of shape (B, hidden_dim) for the configured branch,
        or None if the context branch is disabled."""
        if not self._has_context:
            return None

        if self.bias_replace:
            return self.bias_vec.unsqueeze(0).expand(B, -1)

        parts = []
        if self.use_variety:
            parts.append(self.variety_embed(batch["variety_idx"].squeeze(-1)))
        if self.use_tissue:
            parts.append(self.tissue_embed(batch["tissue_idx"].squeeze(-1)))
        if self.use_stage:
            parts.append(self.stage_embed(batch["stage_idx"].squeeze(-1)))
        parts.append(batch["seq_stats"])
        if self.use_interactions:
            if self.use_variety and self.use_tissue:
                parts.append(batch["variety_tissue"])
            if self.use_tissue and self.use_stage:
                parts.append(batch["tissue_stage"])
            if self.use_variety and self.use_stage:
                parts.append(batch["variety_stage"])
        return self.context_fc(torch.cat(parts, dim=1))

    # ------------------------------------------------------------------
    def forward(self, binned, relpos, firstpos, lastpos, gradient, count, cooc,
                seq_stats, variety, tissue, stage,
                variety_idx=None, tissue_idx=None, stage_idx=None,
                variety_tissue=None, tissue_stage=None, variety_stage=None,
                **kwargs):

        device = binned.device
        B = binned.shape[0]

        # Repackage for _build_context_feat (always available; cheap).
        ctx_batch = {
            "variety_idx": variety_idx, "tissue_idx": tissue_idx, "stage_idx": stage_idx,
            "variety_tissue": variety_tissue, "tissue_stage": tissue_stage,
            "variety_stage": variety_stage, "seq_stats": seq_stats,
        }

        # ---------------- Sequence side ----------------
        motif_combined = None
        pos_feat = None
        if self.use_sequence:
            parts = []
            if self._use_binned or self._use_positional:
                pos_feat = self.positional_proc(
                    binned, relpos, firstpos, lastpos, gradient
                )
                parts.append(pos_feat)
            if self._use_cooc:
                parts.append(self.cooc_fc(cooc))
            if self._use_count:
                count_3d = self.count_embed(count.unsqueeze(-1))
                _, (h_n, _) = self.lstm(count_3d)
                count_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
                count_feat = torch.relu(self.count_proj(count_feat))
                parts.append(count_feat)
            motif_combined = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        # ---------------- Context side ----------------
        context_feat = self._build_context_feat(ctx_batch, B, device)

        # ---------------- Fusion ----------------
        if self.use_sequence and context_feat is not None:
            query = self.context_query(context_feat)
            key = self.motif_key(motif_combined)
            value = self.motif_value(motif_combined)
            attn_score = (query * key).sum(dim=-1, keepdim=True) / self.attn_scale
            attn_weight = torch.sigmoid(attn_score)
            context_aware_motif = attn_weight * value
            # Skip-connection from pos_feat; if positional/binned paths are off,
            # the skip disappears and the attention output stands alone.
            if pos_feat is not None:
                motif_final = self.dropout(pos_feat) + context_aware_motif
            else:
                motif_final = self.dropout(context_aware_motif)

            combined = torch.cat([motif_final, context_feat], dim=1)
            gate_weights = self.gate(combined)
            fused = gate_weights * motif_final + (1 - gate_weights) * context_feat
        elif self.use_sequence:
            # Sequence-only path: project motif_combined into hidden_dim space
            # via the motif_value projection (re-used as the head input).
            value = self.motif_value(motif_combined)
            if pos_feat is not None:
                fused = self.dropout(pos_feat) + value
            else:
                fused = self.dropout(value)
        elif context_feat is not None:
            # Context-only path
            fused = context_feat
        else:
            raise RuntimeError("Both sequence and context branches are disabled — "
                               "nothing to feed the head.")

        return torch.sigmoid(self.output(fused)).squeeze()


# =============================================================================
# CONFIG REGISTRY
# =============================================================================

ABLATION_CONFIGS = [
    {
        "id": 1, "name": "full",
        "label": "Full CARCM (sequence + variety + tissue + stage)",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
    },
    {
        "id": 2, "name": "sequence_only",
        "label": "Sequence-only (context branch removed)",
        "use_sequence": True, "context_dims": [],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
    },
    {
        "id": 3, "name": "seq_variety",
        "label": "Sequence + variety only",
        "use_sequence": True, "context_dims": ["variety"],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
    },
    {
        "id": 4, "name": "seq_tissue",
        "label": "Sequence + tissue only",
        "use_sequence": True, "context_dims": ["tissue"],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
    },
    {
        "id": 5, "name": "seq_stage",
        "label": "Sequence + stage only",
        "use_sequence": True, "context_dims": ["stage"],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
    },
    {
        "id": 6, "name": "permuted_labels",
        "label": "Full CARCM with context labels permuted (negative control)",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": True,
    },
    {
        "id": 7, "name": "context_only",
        "label": "Context-only (no sequence features)",
        "use_sequence": False, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
    },
    {
        "id": 8, "name": "param_matched_bias",
        "label": "Full CARCM with context branch replaced by param-matched bias",
        "use_sequence": True, "context_dims": [],
        "use_interactions": False, "bias_replace": True, "permute_context": False,
    },
    # -----------------------------------------------------------------
    # Feature-level leave-one-out ablation (LOO over the 4 sequence-side
    # feature blocks). All keep full context. Reference for ΔAUC is the
    # Full CARCM row above (id=1, name='full').
    # -----------------------------------------------------------------
    {
        "id": 9, "name": "no_binned",
        "label": "Full CARCM minus Binned Motif Counts",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": False, "positional": True,
                         "count": True, "cooccurrence": True},
        "ablation_group": "feature",
    },
    {
        "id": 10, "name": "no_positional",
        "label": "Full CARCM minus Positional Features",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": False,
                         "count": True, "cooccurrence": True},
        "ablation_group": "feature",
    },
    {
        "id": 11, "name": "no_count",
        "label": "Full CARCM minus Global Motif Counts",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": True,
                         "count": False, "cooccurrence": True},
        "ablation_group": "feature",
    },
    {
        "id": 12, "name": "no_cooc",
        "label": "Full CARCM minus Co-occurrence",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": True,
                         "count": True, "cooccurrence": False},
        "ablation_group": "feature",
    },
    # -----------------------------------------------------------------
    # Sequence-only feature LOO — remove context branch first, then drop
    # one sequence-feature block. Reference for ΔAUC is sequence_only
    # (id=2). This isolates each feature's intrinsic contribution when
    # the model cannot lean on context.
    # -----------------------------------------------------------------
    {
        "id": 13, "name": "seq_only_no_binned",
        "label": "Sequence-only minus Binned Motif Counts",
        "use_sequence": True, "context_dims": [],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": False, "positional": True,
                         "count": True, "cooccurrence": True},
        "ablation_group": "feature_seqonly",
    },
    {
        "id": 14, "name": "seq_only_no_positional",
        "label": "Sequence-only minus Positional Features",
        "use_sequence": True, "context_dims": [],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": False,
                         "count": True, "cooccurrence": True},
        "ablation_group": "feature_seqonly",
    },
    {
        "id": 15, "name": "seq_only_no_count",
        "label": "Sequence-only minus Global Motif Counts",
        "use_sequence": True, "context_dims": [],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": True,
                         "count": False, "cooccurrence": True},
        "ablation_group": "feature_seqonly",
    },
    {
        "id": 16, "name": "seq_only_no_cooc",
        "label": "Sequence-only minus Co-occurrence",
        "use_sequence": True, "context_dims": [],
        "use_interactions": False, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": True,
                         "count": True, "cooccurrence": False},
        "ablation_group": "feature_seqonly",
    },
]


# Helper: which group a config belongs to. The original 8 are 'context'.
def config_group(cfg):
    return cfg.get("ablation_group", "context")


def get_config(name):
    for c in ABLATION_CONFIGS:
        if c["name"] == name:
            return c
    raise ValueError(f"Unknown ablation config: {name}. "
                     f"Available: {[c['name'] for c in ABLATION_CONFIGS]}")
