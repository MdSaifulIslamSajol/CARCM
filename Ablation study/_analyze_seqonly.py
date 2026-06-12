import pandas as pd, numpy as np, os
os.chdir("/home/saiful/CARCM/Ablation study")
agg = pd.read_csv('results/aggregate_metrics.csv')
long = pd.read_csv('results/per_seed_long.csv')
delong = pd.read_csv('results/delong_full_pairwise.csv')

print('=== ENSEMBLE AUC + ΔAUC vs Reference + DeLong p ===')
print()
print('--- Full-context regime (LOO vs Full CARCM = 0.8531) ---')
hdr = ['config','per-seed','ensemble','Δ vs Full','DeLong z','DeLong p']
print(f"  {hdr[0]:<25s}{hdr[1]:>10s}{hdr[2]:>10s}{hdr[3]:>12s}{hdr[4]:>10s}{hdr[5]:>14s}")
full_e = float(agg[agg['name']=='full']['auc_ensemble'].iloc[0])
for n in ['full','no_binned','no_positional','no_count','no_cooc']:
    r = agg[agg['name']==n]
    if r.empty: continue
    e = float(r['auc_ensemble'].iloc[0])
    ps = float(r['auc_mean'].iloc[0])
    d = delong[delong['name']==n]
    z = float(d['z'].iloc[0]) if not d.empty else float('nan')
    p = float(d['p'].iloc[0]) if not d.empty else float('nan')
    if np.isfinite(z):
        print(f"  {n:<25s}{ps:>10.4f}{e:>10.4f}{e-full_e:>+12.4f}{z:>+10.3f}{p:>14.3e}")
    else:
        print(f"  {n:<25s}{ps:>10.4f}{e:>10.4f}{e-full_e:>+12.4f}{'--':>10s}{'--':>14s}")

print()
print('--- No-context regime (LOO vs sequence_only) ---')
print(f"  {hdr[0]:<25s}{hdr[1]:>10s}{hdr[2]:>10s}{'Δ vs SO':>12s}{hdr[4]:>10s}{hdr[5]:>14s}")
seq_e = float(agg[agg['name']=='sequence_only']['auc_ensemble'].iloc[0])
for n in ['sequence_only','seq_only_no_binned','seq_only_no_positional','seq_only_no_count','seq_only_no_cooc']:
    r = agg[agg['name']==n]
    if r.empty: continue
    e = float(r['auc_ensemble'].iloc[0])
    ps = float(r['auc_mean'].iloc[0])
    d = delong[delong['name']==n]
    z = float(d['z'].iloc[0]) if not d.empty else float('nan')
    p = float(d['p'].iloc[0]) if not d.empty else float('nan')
    if np.isfinite(z):
        print(f"  {n:<25s}{ps:>10.4f}{e:>10.4f}{e-seq_e:>+12.4f}{z:>+10.3f}{p:>14.3e}")
    else:
        print(f"  {n:<25s}{ps:>10.4f}{e:>10.4f}{e-seq_e:>+12.4f}{'--':>10s}{'--':>14s}")

print()
print('=== Paired per-seed ΔAUC vs sequence_only (no-context regime) ===')
print(f"  {'config':<25s}{'mean Δ':>10s}{'std':>8s}{'min':>8s}{'max':>8s}{'< 0?':>6s}")
seq_long = long[long['config']=='sequence_only'].sort_values('seed').reset_index(drop=True)
for n in ['seq_only_no_binned','seq_only_no_positional','seq_only_no_count','seq_only_no_cooc']:
    other = long[long['config']==n].sort_values('seed').reset_index(drop=True)
    n_min = min(len(seq_long), len(other))
    diff = other['auc'].values[:n_min] - seq_long['auc'].values[:n_min]
    n_neg = int((diff < 0).sum())
    print(f"  {n:<25s}{diff.mean():>+10.4f}{diff.std(ddof=1):>8.4f}{diff.min():>+8.4f}{diff.max():>+8.4f}{n_neg:>4d}/{n_min}")
