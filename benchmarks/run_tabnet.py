#!/usr/bin/env python3
"""TabNet benchmark on V7 feature matrix."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from common import load_data, compute_metrics, save_result

SEED = 42


def main():
    data = load_data(scale="standard")
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5,
        gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(step_size=20, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        seed=SEED, verbose=1,
    )
    clf.fit(
        X_train=data["X_train"], y_train=data["y_train"],
        eval_set=[(data["X_dev"], data["y_dev"])],
        eval_name=["dev"], eval_metric=["auc"],
        max_epochs=200, patience=25,
        batch_size=1024, virtual_batch_size=128,
        num_workers=2, drop_last=False,
    )
    y_prob = clf.predict_proba(data["X_test"])[:, 1]
    res = compute_metrics(
        data["y_test"], y_prob, data["test_hard_mask"], model_name="TabNet"
    )
    save_result(res, data["y_test"], y_prob, data["test_hard_mask"])


if __name__ == "__main__":
    main()
