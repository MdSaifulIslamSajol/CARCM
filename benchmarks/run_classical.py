#!/usr/bin/env python3
"""Classical same-feature baselines: LR, SVM-RBF, RF, XGBoost, LightGBM, CatBoost.

All models ingest the same 621-dim feature matrix that V7 uses.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
import time
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from common import load_data, compute_metrics, save_result

SEED = 42


def run_model(name, model, data, proba_fn=None):
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"])
    fit_time = time.time() - t0
    if proba_fn is None:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(data["X_test"])[:, 1]
        else:
            from scipy.special import expit
            y_prob = expit(model.decision_function(data["X_test"]))
    else:
        y_prob = proba_fn(model, data["X_test"])
    res = compute_metrics(
        data["y_test"], y_prob, data["test_hard_mask"], model_name=name
    )
    res["fit_time_sec"] = float(fit_time)
    save_result(res, data["y_test"], y_prob, data["test_hard_mask"])
    return res


def main(models):
    data = load_data(scale="standard")
    print(f"Train: {data['X_train'].shape}  Test: {data['X_test'].shape}  "
          f"Hard in test: {data['test_hard_mask'].sum()}")

    registry = {
        "LogisticRegression": lambda: LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", n_jobs=-1,
            random_state=SEED,
        ),
        "SVM_RBF": lambda: _svm(),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=SEED,
        ),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, tree_method="hist",
            eval_metric="auc", early_stopping_rounds=30,
            n_jobs=-1, random_state=SEED,
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=2000, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, n_jobs=-1, random_state=SEED, verbose=-1,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=2000, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=SEED,
            eval_metric="AUC", od_type="Iter", od_wait=50,
            verbose=0,
        ),
    }

    want = list(registry) if models == ["all"] else models
    for name in want:
        if name not in registry:
            print(f"Unknown model {name}; skipping")
            continue
        print(f"\n===== {name} =====")
        model = registry[name]()
        data_i = data
        if name == "XGBoost":
            model.fit(
                data["X_train"], data["y_train"],
                eval_set=[(data["X_dev"], data["y_dev"])],
                verbose=False,
            )
            y_prob = model.predict_proba(data["X_test"])[:, 1]
            res = compute_metrics(
                data["y_test"], y_prob, data["test_hard_mask"], model_name=name
            )
            save_result(res, data["y_test"], y_prob, data["test_hard_mask"])
        elif name == "LightGBM":
            model.fit(
                data["X_train"], data["y_train"],
                eval_set=[(data["X_dev"], data["y_dev"])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            y_prob = model.predict_proba(data["X_test"])[:, 1]
            res = compute_metrics(
                data["y_test"], y_prob, data["test_hard_mask"], model_name=name
            )
            save_result(res, data["y_test"], y_prob, data["test_hard_mask"])
        elif name == "CatBoost":
            model.fit(
                data["X_train"], data["y_train"],
                eval_set=(data["X_dev"], data["y_dev"]),
                verbose=0,
            )
            y_prob = model.predict_proba(data["X_test"])[:, 1]
            res = compute_metrics(
                data["y_test"], y_prob, data["test_hard_mask"], model_name=name
            )
            save_result(res, data["y_test"], y_prob, data["test_hard_mask"])
        else:
            run_model(name, model, data_i)


def _svm():
    # Full RBF SVC on 36k x 621 is slow (> 1h). Use a stratified 15k subsample
    # (fixed seed) for tractable training while preserving class ratio.
    return _SVMWrapper()


class _SVMWrapper:
    def __init__(self):
        self.model = SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True,
            cache_size=1024, random_state=SEED,
        )

    def fit(self, X, y):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rng = np.random.default_rng(SEED)
        n_per_class = 7500
        pos_s = rng.choice(pos, size=min(n_per_class, len(pos)), replace=False)
        neg_s = rng.choice(neg, size=min(n_per_class, len(neg)), replace=False)
        idx = np.concatenate([pos_s, neg_s])
        rng.shuffle(idx)
        self.model.fit(X[idx], y[idx])
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    args = ap.parse_args()
    main(args.models)
