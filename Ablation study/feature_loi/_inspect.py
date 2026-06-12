import pandas as pd, os
os.chdir("/home/saiful/CARCM/Ablation study/feature_loi")
agg = pd.read_csv('results/loi_aggregate_metrics.csv')
keep = ["full","only_binned","only_positional","only_count","only_cooc","sequence_only","context_only"]
v = agg[agg["name"].isin(keep)].copy()
order = {n:i for i,n in enumerate(keep)}
v["__o"] = v["name"].map(order)
v = v.sort_values("__o").reset_index(drop=True)
print('=== LOI summary ===')
print(f"{'config':<22s}{'per-seed':>10s}{'ensemble':>10s}{'Δ_full':>10s}{'Δ_ctxo':>10s}{'p_full':>12s}{'p_ctxo':>12s}")
for _, r in v.iterrows():
    pf = r.get('delong_p_vs_full', float('nan'))
    pc = r.get('delong_p_vs_context_only', float('nan'))
    print(f"  {r['name']:<20s}{r['auc_mean']:>10.4f}{r['auc_ensemble']:>10.4f}{r.get('delta_auc_vs_full_mean', 0):>+10.4f}{r.get('delta_auc_vs_context_only_mean', 0):>+10.4f}{pf:>12.3e}{pc:>12.3e}")
