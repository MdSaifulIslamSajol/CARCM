import pandas as pd, numpy as np
import os
os.chdir("/home/saiful/CARCM/Ablation study")

agg = pd.read_csv('results/aggregate_metrics.csv')
long = pd.read_csv('results/per_seed_long.csv')
delong = pd.read_csv('results/delong_full_pairwise.csv')

print('=== Per-seed dispersion (n=10) ===')
print('config                     mean      std     min      max     range')
for n in ['full','sequence_only','no_binned','no_positional','no_count','no_cooc']:
    s = long[long['config']==n]['auc']
    print(f'  {n:<22s}  {s.mean():.4f}  {s.std(ddof=1):.4f}  {s.min():.4f}  {s.max():.4f}  {s.max()-s.min():.4f}')

print()
print('=== Ensemble AUC + DeLong p vs Full ===')
full_e = float(agg[agg['name']=='full']['auc_ensemble'].iloc[0])
hdr = ['config','ensemble','d_vs_full','DeLong z','DeLong p']
print(f"  {hdr[0]:<20s}{hdr[1]:>10s}{hdr[2]:>12s}{hdr[3]:>10s}{hdr[4]:>14s}")
for n in ['full','sequence_only','no_binned','no_positional','no_count','no_cooc']:
    e = float(agg[agg['name']==n]['auc_ensemble'].iloc[0])
    d = delong[delong['name']==n]
    z = float(d['z'].iloc[0]) if not d.empty else float('nan')
    p = float(d['p'].iloc[0]) if not d.empty else float('nan')
    if np.isfinite(z):
        print(f"  {n:<20s}{e:>10.4f}{e-full_e:>+12.4f}{z:>+10.3f}{p:>14.3e}")
    else:
        print(f"  {n:<20s}{e:>10.4f}{e-full_e:>+12.4f}{'--':>10s}{'--':>14s}")

print()
print('=== Ensemble selection — per-config best method (test AUC by method) ===')
sel = pd.read_csv('results/ensemble_selection.csv')
print(sel.to_string(index=False, float_format='%.4f'))

print()
print('=== Per-seed paired difference vs Full (LOO − Full) ===')
print('config              mean_diff  std_diff   min      max     n_seeds')
full_long = long[long['config']=='full'].sort_values('seed').reset_index(drop=True)
for n in ['no_binned','no_positional','no_count','no_cooc']:
    other = long[long['config']==n].sort_values('seed').reset_index(drop=True)
    n_min = min(len(full_long), len(other))
    diff = other['auc'].values[:n_min] - full_long['auc'].values[:n_min]
    print(f'  {n:<18s}  {diff.mean():>+8.4f}  {diff.std(ddof=1):.4f}  {diff.min():>+7.4f}  {diff.max():>+7.4f}  {n_min}')
