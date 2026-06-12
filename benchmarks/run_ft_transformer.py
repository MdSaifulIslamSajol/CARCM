#!/usr/bin/env python3
"""FT-Transformer benchmark on V7 feature matrix.

Minimal self-contained implementation (numerical-only path: every feature is
treated as numerical, since all 621 features in the V7 matrix are numeric /
one-hot scalars). Follows Gorishniy et al., 'Revisiting Deep Learning Models
for Tabular Data' (NeurIPS 2021).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import load_data, compute_metrics, save_result

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumericalTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):  # x: (B, F)
        return x.unsqueeze(-1) * self.weight + self.bias  # (B, F, d)


class FTBlock(nn.Module):
    def __init__(self, d_token, n_heads, d_ffn, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        h = self.ffn(self.norm2(x))
        x = x + self.drop2(h)
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=64, n_blocks=3, n_heads=8,
                 d_ffn=128, dropout=0.2, n_classes=1):
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            [FTBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        x = self.tokenizer(x)  # (B, F, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x).squeeze(-1)


def train_one(seed, data, epochs=60, patience=10, bs=512, lr=1e-3, wd=1e-5):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(data["X_train"]).float()
    ytr = torch.from_numpy(data["y_train"]).float()
    Xdv = torch.from_numpy(data["X_dev"]).float()
    ydv = torch.from_numpy(data["y_dev"]).float()
    Xte = torch.from_numpy(data["X_test"]).float()

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True,
                        num_workers=2, drop_last=False)

    model = FTTransformer(n_features=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, wait = -1, None, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            dv_pred = torch.sigmoid(model(Xdv.to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(ydv.numpy(), dv_pred)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        print(f"[FT seed={seed}] epoch {ep+1:03d} dev_auc={auc:.4f} best={best_auc:.4f}")
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_prob = torch.sigmoid(model(Xte.to(DEVICE))).cpu().numpy()
    return te_prob, best_auc


def main():
    data = load_data(scale="standard")
    probs = []
    for seed in [42, 123, 456]:
        p, _ = train_one(seed, data)
        probs.append(p)
    y_prob = np.mean(probs, axis=0)
    res = compute_metrics(
        data["y_test"], y_prob, data["test_hard_mask"], model_name="FTTransformer"
    )
    save_result(res, data["y_test"], y_prob, data["test_hard_mask"])


if __name__ == "__main__":
    main()
