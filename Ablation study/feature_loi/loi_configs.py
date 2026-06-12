"""
Leave-one-in (LOI) feature configurations.

Each row keeps ONE sequence-side feature block plus the full
(Variety, Tissue, Stage) context branch. The reference rows for ΔAUC are
the parent study's `full` (all 4 features + ctx, AUC≈0.853) and `context_only`
(no sequence features at all, AUC≈0.681). Trained predictions for those
references already live in `../results/{full,context_only}/`, so we do NOT
retrain them here; we only run the four single-feature configurations.
"""

LOI_CONFIGS = [
    {
        "id": 17, "name": "only_binned",
        "label": "Context + Binned Motif Counts only",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": True, "positional": False,
                         "count": False, "cooccurrence": False},
        "ablation_group": "feature_loi",
    },
    {
        "id": 18, "name": "only_positional",
        "label": "Context + Positional Features only",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": False, "positional": True,
                         "count": False, "cooccurrence": False},
        "ablation_group": "feature_loi",
    },
    {
        "id": 19, "name": "only_count",
        "label": "Context + Global Motif Counts only",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": False, "positional": False,
                         "count": True, "cooccurrence": False},
        "ablation_group": "feature_loi",
    },
    {
        "id": 20, "name": "only_cooc",
        "label": "Context + Co-occurrence only",
        "use_sequence": True, "context_dims": ["variety", "tissue", "stage"],
        "use_interactions": True, "bias_replace": False, "permute_context": False,
        "feature_mask": {"binned": False, "positional": False,
                         "count": False, "cooccurrence": True},
        "ablation_group": "feature_loi",
    },
]


def get_loi_config(name):
    for c in LOI_CONFIGS:
        if c["name"] == name:
            return c
    raise ValueError(f"Unknown LOI config: {name}. "
                     f"Available: {[c['name'] for c in LOI_CONFIGS]}")
