#!/usr/bin/env python3
"""Simple classical baselines for Q1 completeness:
  - Decision Tree
  - Gaussian Naive Bayes
  - Bernoulli Naive Bayes (on binarized features — better fit for motif counts)
  - k-Nearest Neighbors

All ingest the same 621-dim V7 feature matrix.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
import time
import argparse
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from common import load_data, compute_metrics, save_result

SEED = 42


def run(name, model, data):
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"])
    fit_time = time.time() - t0
    y_prob = model.predict_proba(data["X_test"])[:, 1]
    res = compute_metrics(
        data["y_test"], y_prob, data["test_hard_mask"], model_name=name
    )
    res["fit_time_sec"] = float(fit_time)
    save_result(res, data["y_test"], y_prob, data["test_hard_mask"])


def main(models):
    data = load_data(scale="standard")
    print(f"Train: {data['X_train'].shape}  Test: {data['X_test'].shape}  "
          f"Hard in test: {data['test_hard_mask'].sum()}")

    registry = {
        "DecisionTree": lambda: DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=5, random_state=SEED,
        ),
        "GaussianNB": lambda: GaussianNB(),
        "BernoulliNB": lambda: BernoulliNB(alpha=1.0),
        "KNN": lambda: KNeighborsClassifier(
            n_neighbors=25, weights="distance", n_jobs=-1,
        ),
    }

    want = list(registry) if models == ["all"] else models
    for name in want:
        if name not in registry:
            print(f"Unknown {name}; skipping"); continue
        print(f"\n===== {name} =====")
        model = registry[name]()
        # Bernoulli NB wants non-negative / binarized input; bypass the scaled
        # numeric features and use raw counts instead.
        if name == "BernoulliNB":
            raw = load_data(scale="none")
            Xtr = (raw["X_train"] > 0).astype(np.float32)
            Xte = (raw["X_test"] > 0).astype(np.float32)
            model.fit(Xtr, raw["y_train"])
            y_prob = model.predict_proba(Xte)[:, 1]
            res = compute_metrics(
                raw["y_test"], y_prob, raw["test_hard_mask"], model_name=name
            )
            save_result(res, raw["y_test"], y_prob, raw["test_hard_mask"])
        else:
            run(name, model, data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    args = ap.parse_args()
    main(args.models)
