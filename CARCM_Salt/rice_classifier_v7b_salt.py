#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rice Promoter Classifier V7b - SALT / SALINITY adaptation
======================================================================================
Salt-adapted copy of CARCM's rice_classifier_v7b.py. The model architecture and
the training / ensemble / plotting logic are UNCHANGED from the drought version;
only the data paths, the motif count (n_motifs derived from the 47 salinity
motifs instead of a hard-coded 54), and a configurable --gpu id differ. This
keeps the salt-vs-drought comparison on an identical classifier.

Generates all plots, figures, and detailed results for Q1 journal publication.

Key hyperparameters explored:
- Learning rate: [1e-4, 3e-4, 5e-4, 1e-3]
- Hidden dimension: [64, 128, 256]
- Dropout: [0.3, 0.4, 0.5]
- Number of ensemble seeds: [5, 10, 15]
- Batch size: [64, 128, 256]

@author: saiful
"""

import os
import sys
import argparse

# GPU Configuration (configurable via --gpu / CARCM_SALT_GPU; default 0)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CARCM_SALT_GPU", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
import json
import copy
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='V7b Hyperparameter Search')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (default: 0.4)')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Number of ensemble seeds (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='Number of epochs (default: 150)')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience (default: 25)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name (default: auto-generated)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='CUDA device id (default: 0). Note: must be set '
                             'before torch import via CARCM_SALT_GPU env var to '
                             'take effect; this flag is informational.')
    return parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================
# DATA_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/output/"
# CHECKPOINT_DIR = "/media/exxact1/saiful/DeepRice/Checkpoints/binbased_v2/"
# RESULTS_BASE_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/models_v7b/"

DATA_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/output/"
CHECKPOINT_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/checkpoints/"
RESULTS_BASE_DIR = "/home/saiful/DeepRice_Salt/CARCM_Salt/models_carcm_salt/"


# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Salinity motif names (loaded from the pipeline's motif_definitions.json so the
# classifier always matches the feature set produced by the positional pipeline).
try:
    with open(os.path.join(DATA_DIR, 'motif_definitions.json')) as _f:
        MOTIF_NAMES = list(json.load(_f).keys())
except FileNotFoundError:
    MOTIF_NAMES = []  # filled dynamically from feature_groups at run time

NUM_BINS = 5


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_feature_groups(data_dir):
    with open(os.path.join(data_dir, 'feature_groups.json'), 'r') as f:
        return json.load(f)


def create_context_interactions(df, variety_cols, tissue_cols, stage_cols):
    variety_idx = df[variety_cols].values.argmax(axis=1)
    tissue_idx = df[tissue_cols].values.argmax(axis=1)
    stage_idx = df[stage_cols].values.argmax(axis=1)
    
    n_tissues = len(tissue_cols)
    n_stages = len(stage_cols)
    
    vt_idx = variety_idx * n_tissues + tissue_idx
    vt_unique = np.unique(vt_idx)
    vt_map = {v: i for i, v in enumerate(vt_unique)}
    vt_mapped = np.array([vt_map[v] for v in vt_idx])
    vt_onehot = np.zeros((len(df), len(vt_unique)), dtype=np.float32)
    vt_onehot[np.arange(len(df)), vt_mapped] = 1
    
    ts_idx = tissue_idx * n_stages + stage_idx
    ts_unique = np.unique(ts_idx)
    ts_map = {v: i for i, v in enumerate(ts_unique)}
    ts_mapped = np.array([ts_map[v] for v in ts_idx])
    ts_onehot = np.zeros((len(df), len(ts_unique)), dtype=np.float32)
    ts_onehot[np.arange(len(df)), ts_mapped] = 1
    
    vs_idx = variety_idx * n_stages + stage_idx
    vs_unique = np.unique(vs_idx)
    vs_map = {v: i for i, v in enumerate(vs_unique)}
    vs_mapped = np.array([vs_map[v] for v in vs_idx])
    vs_onehot = np.zeros((len(df), len(vs_unique)), dtype=np.float32)
    vs_onehot[np.arange(len(df)), vs_mapped] = 1
    
    return {
        'variety_tissue': vt_onehot, 'n_vt': len(vt_unique),
        'tissue_stage': ts_onehot, 'n_ts': len(ts_unique),
        'variety_stage': vs_onehot, 'n_vs': len(vs_unique),
        'variety_idx': variety_idx,
        'tissue_idx': tissue_idx,
        'stage_idx': stage_idx
    }


def identify_hard_examples(train_df, test_df, count_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_counts = np.round(train_df[count_cols].values, 4)
    train_hashes = [hash(tuple(row)) for row in train_counts]
    train_df['motif_hash'] = train_hashes
    
    conflicting = train_df.groupby('motif_hash')['label'].nunique()
    hard_promoters = set(conflicting[conflicting > 1].index)
    
    train_hard = train_df['motif_hash'].isin(hard_promoters).values
    
    test_counts = np.round(test_df[count_cols].values, 4)
    test_hashes = [hash(tuple(row)) for row in test_counts]
    test_df['motif_hash'] = test_hashes
    test_hard = test_df['motif_hash'].isin(hard_promoters).values
    
    return train_hard, test_hard, hard_promoters


class PositionalMotifDataset(Dataset):
    def __init__(self, df, feature_groups, variety_cols, tissue_cols, stage_cols,
                 context_info=None, hard_examples=None,
                 fit_scaler=False, scaler=None):
        
        binned_cols = feature_groups['binned']
        relpos_cols = feature_groups['relative_position']
        firstpos_cols = feature_groups['first_position']
        lastpos_cols = feature_groups['last_position']
        gradient_cols = feature_groups['density_gradient']
        count_cols = feature_groups['global_count']
        cooc_cols = feature_groups['cooccurrence']
        seq_stat_cols = feature_groups['seq_stats']
        
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
        self.labels = df['label'].values.astype(np.float32)
        
        if context_info is not None:
            self.variety_tissue = context_info['variety_tissue']
            self.tissue_stage = context_info['tissue_stage']
            self.variety_stage = context_info['variety_stage']
            self.variety_idx = context_info['variety_idx']
            self.tissue_idx = context_info['tissue_idx']
            self.stage_idx = context_info['stage_idx']
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
            
            n_binned = self.binned_data.shape[1]
            n_count = self.count_data.shape[1]
            n_cooc = self.cooc_data.shape[1]
            
            self.binned_data = scalable_scaled[:, :n_binned].astype(np.float32)
            self.count_data = scalable_scaled[:, n_binned:n_binned+n_count].astype(np.float32)
            self.cooc_data = scalable_scaled[:, n_binned+n_count:n_binned+n_count+n_cooc].astype(np.float32)
            self.seq_stats = scalable_scaled[:, n_binned+n_count+n_cooc:].astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            scalable = np.hstack([self.binned_data, self.count_data, 
                                  self.cooc_data, self.seq_stats])
            scalable_scaled = self.scaler.transform(scalable)
            
            n_binned = self.binned_data.shape[1]
            n_count = self.count_data.shape[1]
            n_cooc = self.cooc_data.shape[1]
            
            self.binned_data = scalable_scaled[:, :n_binned].astype(np.float32)
            self.count_data = scalable_scaled[:, n_binned:n_binned+n_count].astype(np.float32)
            self.cooc_data = scalable_scaled[:, n_binned+n_count:n_binned+n_count+n_cooc].astype(np.float32)
            self.seq_stats = scalable_scaled[:, n_binned+n_count+n_cooc:].astype(np.float32)
        else:
            self.scaler = None
        
        self.n_binned = self.binned_data.shape[1]
        self.n_relpos = self.relpos_data.shape[1]
        self.n_cooc = self.cooc_data.shape[1]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'binned': torch.FloatTensor(self.binned_data[idx]),
            'relpos': torch.FloatTensor(self.relpos_data[idx]),
            'firstpos': torch.FloatTensor(self.firstpos_data[idx]),
            'lastpos': torch.FloatTensor(self.lastpos_data[idx]),
            'gradient': torch.FloatTensor(self.gradient_data[idx]),
            'count': torch.FloatTensor(self.count_data[idx]),
            'cooc': torch.FloatTensor(self.cooc_data[idx]),
            'seq_stats': torch.FloatTensor(self.seq_stats[idx]),
            'variety': torch.FloatTensor(self.variety_data[idx]),
            'tissue': torch.FloatTensor(self.tissue_data[idx]),
            'stage': torch.FloatTensor(self.stage_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]]),
            'is_hard': torch.FloatTensor([self.is_hard[idx]])
        }
        
        if self.variety_tissue is not None:
            item['variety_tissue'] = torch.FloatTensor(self.variety_tissue[idx])
            item['tissue_stage'] = torch.FloatTensor(self.tissue_stage[idx])
            item['variety_stage'] = torch.FloatTensor(self.variety_stage[idx])
            item['variety_idx'] = torch.LongTensor([self.variety_idx[idx]])
            item['tissue_idx'] = torch.LongTensor([self.tissue_idx[idx]])
            item['stage_idx'] = torch.LongTensor([self.stage_idx[idx]])
        
        return item


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ContextEmbeddingModule(nn.Module):
    def __init__(self, n_varieties, n_tissues, n_stages, embed_dim=64):
        super().__init__()
        self.variety_embed = nn.Embedding(n_varieties, embed_dim)
        self.tissue_embed = nn.Embedding(n_tissues, embed_dim)
        self.stage_embed = nn.Embedding(n_stages, embed_dim)
        
        nn.init.normal_(self.variety_embed.weight, 0, 0.1)
        nn.init.normal_(self.tissue_embed.weight, 0, 0.1)
        nn.init.normal_(self.stage_embed.weight, 0, 0.1)
    
    def forward(self, variety_idx, tissue_idx, stage_idx):
        return (self.variety_embed(variety_idx.squeeze(-1)),
                self.tissue_embed(tissue_idx.squeeze(-1)),
                self.stage_embed(stage_idx.squeeze(-1)))


class PositionalMotifProcessor(nn.Module):
    def __init__(self, n_motifs, n_bins, hidden_dim, dropout=0.4):
        super().__init__()
        self.n_motifs = n_motifs
        self.n_bins = n_bins
        
        self.bin_conv = nn.Sequential(
            nn.Conv1d(n_motifs, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bin_attention = nn.Sequential(
            nn.Linear(n_bins, n_bins),
            nn.Softmax(dim=-1)
        )
        
        self.pos_fc = nn.Sequential(
            nn.Linear(n_motifs * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, binned, relpos, firstpos, lastpos, gradient):
        batch_size = binned.shape[0]
        
        binned_3d = binned.view(batch_size, self.n_motifs, self.n_bins)
        bin_weights = self.bin_attention(binned_3d)
        binned_weighted = binned_3d * bin_weights
        
        bin_features = self.bin_conv(binned_weighted)
        bin_features = bin_features.mean(dim=2)
        
        pos_concat = torch.cat([relpos, firstpos, lastpos, gradient], dim=1)
        pos_features = self.pos_fc(pos_concat)
        
        combined = torch.cat([bin_features, pos_features], dim=1)
        return self.combine(combined)


class ContextEnhancedGatedFusionV5(nn.Module):
    def __init__(self, n_motifs, n_bins, n_cooc, n_varieties, n_tissues, n_stages,
                 num_seq_stats=4, hidden_dim=128, embed_dim=64, dropout=0.4,
                 n_vt=0, n_ts=0, n_vs=0):
        super().__init__()
        
        self.n_motifs = n_motifs
        self.n_bins = n_bins
        
        self.context_embed = ContextEmbeddingModule(
            n_varieties, n_tissues, n_stages, embed_dim
        )
        
        self.positional_proc = PositionalMotifProcessor(
            n_motifs, n_bins, hidden_dim, dropout
        )
        
        self.cooc_fc = nn.Sequential(
            nn.Linear(n_cooc, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.count_embed = nn.Linear(1, 32)
        self.lstm = nn.LSTM(32, hidden_dim // 2, num_layers=2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.count_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        context_input_dim = embed_dim * 3 + n_vt + n_ts + n_vs + num_seq_stats
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        motif_feature_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2
        self.context_query = nn.Linear(hidden_dim, hidden_dim)
        self.motif_key = nn.Linear(motif_feature_dim, hidden_dim)
        self.motif_value = nn.Linear(motif_feature_dim, hidden_dim)
        self.attn_scale = hidden_dim ** 0.5
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, binned, relpos, firstpos, lastpos, gradient, count, cooc,
                seq_stats, variety, tissue, stage,
                variety_idx=None, tissue_idx=None, stage_idx=None,
                variety_tissue=None, tissue_stage=None, variety_stage=None, **kwargs):
        
        variety_emb, tissue_emb, stage_emb = self.context_embed(
            variety_idx, tissue_idx, stage_idx
        )
        
        pos_feat = self.positional_proc(binned, relpos, firstpos, lastpos, gradient)
        cooc_feat = self.cooc_fc(cooc)
        
        count_3d = self.count_embed(count.unsqueeze(-1))
        _, (h_n, _) = self.lstm(count_3d)
        count_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        count_feat = torch.relu(self.count_proj(count_feat))
        
        motif_combined = torch.cat([pos_feat, cooc_feat, count_feat], dim=1)
        
        context_parts = [variety_emb, tissue_emb, stage_emb, seq_stats]
        if variety_tissue is not None:
            context_parts.extend([variety_tissue, tissue_stage, variety_stage])
        context_vec = torch.cat(context_parts, dim=1)
        context_feat = self.context_fc(context_vec)
        
        query = self.context_query(context_feat)
        key = self.motif_key(motif_combined)
        value = self.motif_value(motif_combined)
        
        attn_score = (query * key).sum(dim=-1, keepdim=True) / self.attn_scale
        attn_weight = torch.sigmoid(attn_score)
        context_aware_motif = attn_weight * value
        
        motif_final = self.dropout(pos_feat) + context_aware_motif
        
        combined = torch.cat([motif_final, context_feat], dim=1)
        gate_weights = self.gate(combined)
        fused = gate_weights * motif_final + (1 - gate_weights) * context_feat
        
        return torch.sigmoid(self.output(fused)).squeeze()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def hard_weighted_bce_loss(pred, target, is_hard, hard_weight=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    weights = 1.0 + is_hard * (hard_weight - 1.0)
    return (bce * weights).mean()


def train_single_model(model, train_loader, dev_loader, device, config, seed):
    """Train a single model and return training history."""
    set_seed(seed)
    
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    best_auc = 0
    best_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [], 'dev_auc': [], 'dev_acc': [], 
        'dev_hard_auc': [], 'lr': []
    }
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = hard_weighted_bce_loss(
                outputs, batch['label'].squeeze(), 
                batch['is_hard'].squeeze(), hard_weight=2.0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Evaluate
        results = evaluate_model(model, dev_loader, device)
        history['dev_auc'].append(results['auc'])
        history['dev_acc'].append(results['acc'])
        history['dev_hard_auc'].append(results['hard_auc'])
        
        scheduler.step(results['auc'])
        
        if results['auc'] > best_auc:
            best_auc = results['auc']
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{config['num_epochs']} | "
                  f"Loss: {avg_loss:.4f} | Dev AUC: {results['auc']:.4f} | "
                  f"Hard AUC: {results['hard_auc']:.4f}")
        
        if patience_counter >= config['patience']:
            print(f"      Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_auc, history


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_hard = []
    
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            outputs = model(**batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch['label'].squeeze().cpu().numpy())
            all_hard.extend(batch['is_hard'].squeeze().cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    hard = np.array(all_hard).astype(bool)
    
    auc_score = roc_auc_score(labels, preds)
    pred_binary = (preds > 0.5).astype(int)
    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary)
    
    easy_mask = ~hard
    results = {
        'auc': auc_score, 'acc': acc, 'f1': f1,
        'easy_auc': roc_auc_score(labels[easy_mask], preds[easy_mask]) if easy_mask.sum() > 10 else 0,
        'hard_auc': roc_auc_score(labels[hard], preds[hard]) if hard.sum() > 10 else 0,
        'easy_n': int(easy_mask.sum()),
        'hard_n': int(hard.sum()),
        'preds': preds,
        'labels': labels,
        'hard_mask': hard
    }
    
    return results


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            outputs = model(**batch)
            all_preds.extend(outputs.cpu().numpy())
    
    return np.array(all_preds)


# =============================================================================
# PUBLICATION-QUALITY PLOTTING FUNCTIONS
# =============================================================================

def plot_roc_curves(labels, preds, hard_mask, output_dir, config_name):
    """Plot ROC curves for overall, easy, and hard examples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Overall ROC
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.2)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Overall ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Easy examples ROC
    easy_mask = ~hard_mask
    if easy_mask.sum() > 10:
        fpr_easy, tpr_easy, _ = roc_curve(labels[easy_mask], preds[easy_mask])
        roc_auc_easy = auc(fpr_easy, tpr_easy)
        axes[1].plot(fpr_easy, tpr_easy, 'g-', lw=2, label=f'AUC = {roc_auc_easy:.4f}')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].fill_between(fpr_easy, tpr_easy, alpha=0.2, color='green')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'Easy Examples ROC (n={easy_mask.sum()})')
    axes[1].legend(loc='lower right')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    # Hard examples ROC
    if hard_mask.sum() > 10:
        fpr_hard, tpr_hard, _ = roc_curve(labels[hard_mask], preds[hard_mask])
        roc_auc_hard = auc(fpr_hard, tpr_hard)
        axes[2].plot(fpr_hard, tpr_hard, 'r-', lw=2, label=f'AUC = {roc_auc_hard:.4f}')
        axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[2].fill_between(fpr_hard, tpr_hard, alpha=0.2, color='red')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title(f'Hard Examples ROC (n={hard_mask.sum()})')
    axes[2].legend(loc='lower right')
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_roc_curves.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_roc_curves.pdf'))
    plt.close()


def plot_pr_curves(labels, preds, hard_mask, output_dir, config_name):
    """Plot Precision-Recall curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Overall PR
    precision, recall, _ = precision_recall_curve(labels, preds)
    ap = average_precision_score(labels, preds)
    axes[0].plot(recall, precision, 'b-', lw=2, label=f'AP = {ap:.4f}')
    axes[0].fill_between(recall, precision, alpha=0.2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Overall Precision-Recall Curve')
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Easy examples PR
    easy_mask = ~hard_mask
    if easy_mask.sum() > 10:
        prec_easy, rec_easy, _ = precision_recall_curve(labels[easy_mask], preds[easy_mask])
        ap_easy = average_precision_score(labels[easy_mask], preds[easy_mask])
        axes[1].plot(rec_easy, prec_easy, 'g-', lw=2, label=f'AP = {ap_easy:.4f}')
        axes[1].fill_between(rec_easy, prec_easy, alpha=0.2, color='green')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Easy Examples PR (n={easy_mask.sum()})')
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    # Hard examples PR
    if hard_mask.sum() > 10:
        prec_hard, rec_hard, _ = precision_recall_curve(labels[hard_mask], preds[hard_mask])
        ap_hard = average_precision_score(labels[hard_mask], preds[hard_mask])
        axes[2].plot(rec_hard, prec_hard, 'r-', lw=2, label=f'AP = {ap_hard:.4f}')
        axes[2].fill_between(rec_hard, prec_hard, alpha=0.2, color='red')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title(f'Hard Examples PR (n={hard_mask.sum()})')
    axes[2].legend(loc='upper right')
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_pr_curves.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_pr_curves.pdf'))
    plt.close()


def plot_confusion_matrix(labels, preds, threshold, output_dir, config_name):
    """Plot confusion matrix."""
    pred_binary = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_binary)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                annot_kws={'size': 16}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})', fontsize=14)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=12, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_confusion_matrix.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_confusion_matrix.pdf'))
    plt.close()


def plot_threshold_analysis(labels, preds, output_dir, config_name):
    """Plot metrics at different thresholds."""
    thresholds = np.arange(0.1, 0.91, 0.01)
    metrics = {'threshold': [], 'accuracy': [], 'precision': [], 
               'recall': [], 'f1': [], 'specificity': []}
    
    for t in thresholds:
        pred_binary = (preds >= t).astype(int)
        cm = confusion_matrix(labels, pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['threshold'].append(t)
        metrics['accuracy'].append((tp + tn) / (tp + tn + fp + fn))
        metrics['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics['recall'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics['f1'].append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
        metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['threshold'], metrics['accuracy'], 'b-', label='Accuracy', lw=2)
    ax.plot(metrics['threshold'], metrics['precision'], 'g-', label='Precision', lw=2)
    ax.plot(metrics['threshold'], metrics['recall'], 'r-', label='Recall', lw=2)
    ax.plot(metrics['threshold'], metrics['f1'], 'm-', label='F1 Score', lw=2)
    ax.plot(metrics['threshold'], metrics['specificity'], 'c-', label='Specificity', lw=2)
    
    # Find optimal threshold (max F1)
    best_idx = np.argmax(metrics['f1'])
    best_thresh = metrics['threshold'][best_idx]
    ax.axvline(x=best_thresh, color='gray', linestyle='--', label=f'Optimal (F1={metrics["f1"][best_idx]:.3f})')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics vs. Classification Threshold')
    ax.legend(loc='best')
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_threshold_analysis.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_threshold_analysis.pdf'))
    plt.close()
    
    return best_thresh


def plot_calibration_curve(labels, preds, output_dir, config_name):
    """Plot calibration curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(labels, preds, n_bins=10)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    axes[0].plot(prob_pred, prob_true, 'b-o', label='Model')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curve')
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Histogram of predictions
    axes[1].hist(preds[labels == 0], bins=50, alpha=0.5, label='Down (label=0)', color='blue')
    axes[1].hist(preds[labels == 1], bins=50, alpha=0.5, label='Up (label=1)', color='red')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Predictions')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_calibration.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_calibration.pdf'))
    plt.close()


def plot_training_history(all_histories, output_dir, config_name):
    """Plot training history for all seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    for idx, (seed, hist) in enumerate(all_histories.items()):
        epochs = range(1, len(hist['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, hist['train_loss'], color=colors[idx], 
                        alpha=0.7, label=f'Seed {seed}')
        axes[0, 1].plot(epochs, hist['dev_auc'], color=colors[idx], alpha=0.7)
        axes[1, 0].plot(epochs, hist['dev_hard_auc'], color=colors[idx], alpha=0.7)
        axes[1, 1].plot(epochs, hist['lr'], color=colors[idx], alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dev AUC')
    axes[0, 1].set_title('Validation AUC')
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dev Hard AUC')
    axes[1, 0].set_title('Hard Examples AUC')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_training_history.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_training_history.pdf'))
    plt.close()


def plot_ensemble_comparison(base_aucs, ensemble_results, output_dir, config_name):
    """Plot comparison of base models vs ensemble methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Base model AUCs
    seeds = list(base_aucs.keys())
    aucs = list(base_aucs.values())
    x = range(len(seeds))
    
    axes[0].bar(x, aucs, color='steelblue', alpha=0.8)
    axes[0].axhline(y=np.mean(aucs), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(aucs):.4f}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Seed\n{s}' for s in seeds], fontsize=9)
    axes[0].set_ylabel('Dev AUC')
    axes[0].set_title('Individual Base Model Performance')
    axes[0].legend()
    
    # Ensemble comparison
    methods = list(ensemble_results.keys())
    test_aucs = [ensemble_results[m]['test_auc'] for m in methods]
    hard_aucs = [ensemble_results[m]['hard_auc'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, test_aucs, width, label='Test AUC', color='steelblue')
    bars2 = axes[1].bar(x + width/2, hard_aucs, width, label='Hard AUC', color='coral')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Ensemble Methods Comparison')
    axes[1].legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[1].annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config_name}_ensemble_comparison.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_ensemble_comparison.pdf'))
    plt.close()


def plot_summary_dashboard(results, output_dir, config_name):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(results['labels'], results['preds'])
    ax1.plot(fpr, tpr, 'b-', lw=2, label=f"AUC = {results['test_auc']:.4f}")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.2)
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    
    # PR Curve
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(results['labels'], results['preds'])
    ap = average_precision_score(results['labels'], results['preds'])
    ax2.plot(rec, prec, 'g-', lw=2, label=f'AP = {ap:.4f}')
    ax2.fill_between(rec, prec, alpha=0.2, color='green')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='upper right')
    
    # Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    pred_binary = (results['preds'] >= results['optimal_threshold']).astype(int)
    cm = confusion_matrix(results['labels'], pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title(f"Confusion (t={results['optimal_threshold']:.2f})")
    
    # Metrics bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    metrics = ['AUC', 'Hard AUC', 'Easy AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
    values = [results['test_auc'], results['hard_auc'], results['easy_auc'],
              results['accuracy'], results['f1'], results['precision'], results['recall']]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))
    bars = ax4.barh(metrics, values, color=colors)
    ax4.set_xlim([0, 1])
    ax4.set_title('Performance Metrics')
    for bar, val in zip(bars, values):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=10)
    
    # Prediction distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(results['preds'][results['labels'] == 0], bins=50, alpha=0.5, 
             label='Down', color='blue', density=True)
    ax5.hist(results['preds'][results['labels'] == 1], bins=50, alpha=0.5, 
             label='Up', color='red', density=True)
    ax5.axvline(x=results['optimal_threshold'], color='black', linestyle='--', 
                label=f"Threshold={results['optimal_threshold']:.2f}")
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution')
    ax5.legend()
    
    # Hard vs Easy comparison
    ax6 = fig.add_subplot(gs[1, 2])
    categories = ['Overall', 'Easy', 'Hard']
    auc_values = [results['test_auc'], results['easy_auc'], results['hard_auc']]
    colors = ['steelblue', 'green', 'coral']
    bars = ax6.bar(categories, auc_values, color=colors)
    ax6.set_ylabel('AUC')
    ax6.set_title('AUC by Example Difficulty')
    ax6.set_ylim([0.5, 1.0])
    for bar, val in zip(bars, auc_values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}', 
                ha='center', fontsize=11)
    
    # Configuration text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    config_text = f"""
    Configuration: {config_name}
    
    Hyperparameters:
    • Learning Rate: {results['config']['learning_rate']}
    • Hidden Dimension: {results['config']['hidden_dim']}
    • Dropout: {results['config']['dropout']}
    • Batch Size: {results['config']['batch_size']}
    • Number of Seeds: {results['config']['num_seeds']}
    • Weight Decay: {results['config']['weight_decay']}
    
    Final Results:
    • Test AUC: {results['test_auc']:.4f}
    • Hard AUC: {results['hard_auc']:.4f}
    • Easy AUC: {results['easy_auc']:.4f}
    • Accuracy: {results['accuracy']:.4f}
    • Precision: {results['precision']:.4f}
    • Recall: {results['recall']:.4f}
    • F1 Score: {results['f1']:.4f}
    • MCC: {results['mcc']:.4f}
    • Optimal Threshold: {results['optimal_threshold']:.2f}
    """
    ax7.text(0.5, 0.5, config_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f"V7b Results Summary: {config_name}", fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, f'{config_name}_summary_dashboard.png'))
    plt.savefig(os.path.join(output_dir, f'{config_name}_summary_dashboard.pdf'))
    plt.close()


def save_detailed_results(results, output_dir, config_name):
    """Save detailed results to JSON and CSV."""
    # JSON results
    json_results = {
        'config': results['config'],
        'metrics': {
            'test_auc': float(results['test_auc']),
            'hard_auc': float(results['hard_auc']),
            'easy_auc': float(results['easy_auc']),
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'mcc': float(results['mcc']),
            'optimal_threshold': float(results['optimal_threshold'])
        },
        'base_model_aucs': {str(k): float(v) for k, v in results['base_aucs'].items()},
        'ensemble_results': {k: {kk: float(vv) for kk, vv in v.items()} 
                            for k, v in results['ensemble_results'].items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, f'{config_name}_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # CSV with predictions
    pred_df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_prob': results['preds'],
        'predicted_label': (results['preds'] >= results['optimal_threshold']).astype(int),
        'is_hard': results['hard_mask'].astype(int)
    })
    pred_df.to_csv(os.path.join(output_dir, f'{config_name}_predictions.csv'), index=False)
    
    # Metrics at different thresholds
    threshold_metrics = []
    for t in np.arange(0.1, 0.91, 0.05):
        pred_binary = (results['preds'] >= t).astype(int)
        cm = confusion_matrix(results['labels'], pred_binary)
        tn, fp, fn, tp = cm.ravel()
        threshold_metrics.append({
            'threshold': t,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    pd.DataFrame(threshold_metrics).to_csv(
        os.path.join(output_dir, f'{config_name}_threshold_metrics.csv'), index=False
    )


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_experiment(config):
    """Run complete experiment with given configuration."""
    
    # Create run directory
    if config['run_name']:
        run_name = config['run_name']
    else:
        run_name = f"lr{config['learning_rate']}_hd{config['hidden_dim']}_do{config['dropout']}_bs{config['batch_size']}_ns{config['num_seeds']}"
    
    output_dir = os.path.join(RESULTS_BASE_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Redirect stdout
    log_file = open(os.path.join(output_dir, 'console_output.txt'), 'w')
    original_stdout = sys.stdout
    
    class TeeOutput:
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout
            self.closed = False
        def write(self, data):
            if not self.closed:
                try:
                    self.file.write(data)
                except:
                    pass
            self.stdout.write(data)
        def flush(self):
            if not self.closed:
                try:
                    self.file.flush()
                except:
                    pass
            self.stdout.flush()
        def close(self):
            self.closed = True
    
    tee = TeeOutput(log_file, original_stdout)
    sys.stdout = tee
    
    print("=" * 80)
    print(f"RICE PROMOTER CLASSIFIER V7b - HYPERPARAMETER SEARCH")
    print(f"Run: {run_name}")
    print("=" * 80)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    feature_groups = load_feature_groups(DATA_DIR)
    
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    
    # Get column info
    variety_cols = [c for c in train_df.columns if c.startswith('Variety_')]
    tissue_cols = [c for c in train_df.columns if c.startswith('Tissue_')]
    stage_cols = [c for c in train_df.columns if c.startswith('Stage_')]
    
    # Context interactions
    train_context = create_context_interactions(train_df, variety_cols, tissue_cols, stage_cols)
    dev_context = create_context_interactions(dev_df, variety_cols, tissue_cols, stage_cols)
    test_context = create_context_interactions(test_df, variety_cols, tissue_cols, stage_cols)
    
    # Identify hard examples
    count_cols = feature_groups['global_count']
    train_hard, test_hard, _ = identify_hard_examples(train_df, test_df, count_cols)
    _, dev_hard, _ = identify_hard_examples(train_df, dev_df, count_cols)
    
    print(f"  Train hard examples: {train_hard.sum()} ({train_hard.mean()*100:.1f}%)")
    print(f"  Test hard examples: {test_hard.sum()} ({test_hard.mean()*100:.1f}%)")
    
    # Create datasets
    train_dataset = PositionalMotifDataset(
        train_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=train_context, hard_examples=train_hard, fit_scaler=True
    )
    dev_dataset = PositionalMotifDataset(
        dev_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=dev_context, hard_examples=dev_hard, scaler=train_dataset.scaler
    )
    test_dataset = PositionalMotifDataset(
        test_df, feature_groups, variety_cols, tissue_cols, stage_cols,
        context_info=test_context, hard_examples=test_hard, scaler=train_dataset.scaler
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Model parameters
    # n_motifs derived from the feature set (47 salinity motifs), not hard-coded.
    # global_count has exactly one feature per motif, so it gives the motif count.
    n_motifs = len(feature_groups['global_count'])
    assert len(feature_groups['binned']) == n_motifs * NUM_BINS, (
        f"binned features ({len(feature_groups['binned'])}) != n_motifs*NUM_BINS "
        f"({n_motifs}*{NUM_BINS})")
    n_bins = NUM_BINS
    n_cooc = len(feature_groups['cooccurrence'])
    n_varieties = len(variety_cols)
    n_tissues = len(tissue_cols)
    n_stages = len(stage_cols)
    
    # Generate seeds
    base_seeds = [42 + i * 1000 for i in range(config['num_seeds'])]
    
    print(f"\n{'='*80}")
    print(f"STAGE 1: TRAINING {config['num_seeds']} BASE MODELS")
    print(f"{'='*80}")
    
    # Train base models
    base_models = []
    base_aucs = {}
    all_histories = {}
    
    start_time = time.time()
    
    for idx, seed in enumerate(base_seeds):
        print(f"\n[{idx+1}/{len(base_seeds)}] Training with seed={seed}")
        set_seed(seed)
        
        model = ContextEnhancedGatedFusionV5(
            n_motifs=n_motifs, n_bins=n_bins, n_cooc=n_cooc,
            n_varieties=n_varieties, n_tissues=n_tissues, n_stages=n_stages,
            num_seq_stats=4, hidden_dim=config['hidden_dim'],
            embed_dim=64, dropout=config['dropout'],
            n_vt=train_context['n_vt'], n_ts=train_context['n_ts'], n_vs=train_context['n_vs']
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {n_params:,}")
        
        model, best_auc, history = train_single_model(
            model, train_loader, dev_loader, device, config, seed
        )
        
        base_models.append(model)
        base_aucs[seed] = best_auc
        all_histories[seed] = history
        
        print(f"    Seed {seed}: Dev AUC = {best_auc:.4f}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_seed{seed}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'seed': seed,
            'dev_auc': best_auc
        }, checkpoint_path)
    
    train_time = time.time() - start_time
    print(f"\nTotal training time: {train_time/60:.1f} minutes")
    
    # Plot training history
    plot_training_history(all_histories, output_dir, run_name)
    
    print(f"\n{'='*80}")
    print("STAGE 2: ENSEMBLE METHODS")
    print(f"{'='*80}")
    
    # Get all predictions
    all_test_preds = []
    all_dev_preds = []
    test_labels = None
    dev_labels = None
    
    for model in base_models:
        test_results = evaluate_model(model, test_loader, device)
        dev_results = evaluate_model(model, dev_loader, device)
        all_test_preds.append(test_results['preds'])
        all_dev_preds.append(dev_results['preds'])
        if test_labels is None:
            test_labels = test_results['labels']
            test_hard_mask = test_results['hard_mask']
            dev_labels = dev_results['labels']
    
    all_test_preds = np.array(all_test_preds)
    all_dev_preds = np.array(all_dev_preds)
    
    ensemble_results = {}
    
    # 1. Simple Average
    simple_avg = all_test_preds.mean(axis=0)
    simple_auc = roc_auc_score(test_labels, simple_avg)
    simple_hard_auc = roc_auc_score(test_labels[test_hard_mask], simple_avg[test_hard_mask])
    ensemble_results['simple_avg'] = {'test_auc': simple_auc, 'hard_auc': simple_hard_auc}
    print(f"\n1. Simple Average: Test AUC = {simple_auc:.4f} | Hard AUC = {simple_hard_auc:.4f}")
    
    # 2. Weighted Average
    dev_aucs = np.array(list(base_aucs.values()))
    weights = dev_aucs / dev_aucs.sum()
    weighted_avg = np.average(all_test_preds, axis=0, weights=weights)
    weighted_auc = roc_auc_score(test_labels, weighted_avg)
    weighted_hard_auc = roc_auc_score(test_labels[test_hard_mask], weighted_avg[test_hard_mask])
    ensemble_results['weighted_avg'] = {'test_auc': weighted_auc, 'hard_auc': weighted_hard_auc}
    print(f"2. Weighted Average: Test AUC = {weighted_auc:.4f} | Hard AUC = {weighted_hard_auc:.4f}")
    
    # 3. Rank Average
    rank_preds = np.array([rankdata(p) / len(p) for p in all_test_preds])
    rank_avg = rank_preds.mean(axis=0)
    rank_auc = roc_auc_score(test_labels, rank_avg)
    rank_hard_auc = roc_auc_score(test_labels[test_hard_mask], rank_avg[test_hard_mask])
    ensemble_results['rank_avg'] = {'test_auc': rank_auc, 'hard_auc': rank_hard_auc}
    print(f"3. Rank Average: Test AUC = {rank_auc:.4f} | Hard AUC = {rank_hard_auc:.4f}")
    
    # 4. Meta-Learner
    meta_X_train = all_dev_preds.T
    meta_X_test = all_test_preds.T
    meta_learner = LogisticRegression(C=1.0, max_iter=1000)
    meta_learner.fit(meta_X_train, dev_labels)
    meta_preds = meta_learner.predict_proba(meta_X_test)[:, 1]
    meta_auc = roc_auc_score(test_labels, meta_preds)
    meta_hard_auc = roc_auc_score(test_labels[test_hard_mask], meta_preds[test_hard_mask])
    ensemble_results['meta_learner'] = {'test_auc': meta_auc, 'hard_auc': meta_hard_auc}
    print(f"4. Meta-Learner: Test AUC = {meta_auc:.4f} | Hard AUC = {meta_hard_auc:.4f}")
    
    # 5. Blended
    blended = 0.4 * weighted_avg + 0.3 * rank_avg + 0.3 * meta_preds
    blended_auc = roc_auc_score(test_labels, blended)
    blended_hard_auc = roc_auc_score(test_labels[test_hard_mask], blended[test_hard_mask])
    ensemble_results['blended'] = {'test_auc': blended_auc, 'hard_auc': blended_hard_auc}
    print(f"5. Blended: Test AUC = {blended_auc:.4f} | Hard AUC = {blended_hard_auc:.4f}")
    
    # Find best method
    best_method = max(ensemble_results, key=lambda x: ensemble_results[x]['test_auc'])
    
    if best_method == 'simple_avg':
        final_preds = simple_avg
    elif best_method == 'weighted_avg':
        final_preds = weighted_avg
    elif best_method == 'rank_avg':
        final_preds = rank_avg
    elif best_method == 'meta_learner':
        final_preds = meta_preds
    else:
        final_preds = blended
    
    print(f"\n✓ Best Method: {best_method}")
    
    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        pred_binary = (final_preds >= t).astype(int)
        f1 = f1_score(test_labels, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    
    # Compute final metrics
    pred_binary = (final_preds >= best_thresh).astype(int)
    easy_mask = ~test_hard_mask
    
    final_results = {
        'config': config,
        'run_name': run_name,
        'test_auc': roc_auc_score(test_labels, final_preds),
        'hard_auc': roc_auc_score(test_labels[test_hard_mask], final_preds[test_hard_mask]),
        'easy_auc': roc_auc_score(test_labels[easy_mask], final_preds[easy_mask]),
        'accuracy': accuracy_score(test_labels, pred_binary),
        'precision': precision_score(test_labels, pred_binary),
        'recall': recall_score(test_labels, pred_binary),
        'f1': f1_score(test_labels, pred_binary),
        'mcc': matthews_corrcoef(test_labels, pred_binary),
        'optimal_threshold': best_thresh,
        'labels': test_labels,
        'preds': final_preds,
        'hard_mask': test_hard_mask,
        'base_aucs': base_aucs,
        'ensemble_results': ensemble_results,
        'best_method': best_method,
        'training_time_minutes': train_time / 60
    }
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\n  Test AUC:      {final_results['test_auc']:.4f}")
    print(f"  Hard AUC:      {final_results['hard_auc']:.4f}")
    print(f"  Easy AUC:      {final_results['easy_auc']:.4f}")
    print(f"  Accuracy:      {final_results['accuracy']:.4f}")
    print(f"  Precision:     {final_results['precision']:.4f}")
    print(f"  Recall:        {final_results['recall']:.4f}")
    print(f"  F1 Score:      {final_results['f1']:.4f}")
    print(f"  MCC:           {final_results['mcc']:.4f}")
    print(f"  Threshold:     {final_results['optimal_threshold']:.2f}")
    
    # Generate all plots
    print(f"\n{'='*80}")
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print(f"{'='*80}")
    
    print("\n  Generating ROC curves...")
    plot_roc_curves(test_labels, final_preds, test_hard_mask, output_dir, run_name)
    
    print("  Generating PR curves...")
    plot_pr_curves(test_labels, final_preds, test_hard_mask, output_dir, run_name)
    
    print("  Generating confusion matrix...")
    plot_confusion_matrix(test_labels, final_preds, best_thresh, output_dir, run_name)
    
    print("  Generating threshold analysis...")
    plot_threshold_analysis(test_labels, final_preds, output_dir, run_name)
    
    print("  Generating calibration curve...")
    plot_calibration_curve(test_labels, final_preds, output_dir, run_name)
    
    print("  Generating ensemble comparison...")
    plot_ensemble_comparison(base_aucs, ensemble_results, output_dir, run_name)
    
    print("  Generating summary dashboard...")
    plot_summary_dashboard(final_results, output_dir, run_name)
    
    print("  Saving detailed results...")
    save_detailed_results(final_results, output_dir, run_name)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {run_name}")
    print(f"{'='*80}")
    print(f"\n  Results saved to: {output_dir}")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    
    # Restore stdout before closing log file
    tee.close()
    sys.stdout = original_stdout
    log_file.close()
    
    return final_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    config = {
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'num_seeds': args.num_seeds,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'run_name': args.run_name
    }
    
    results = run_experiment(config)
    
    # Results already printed inside run_experiment()
