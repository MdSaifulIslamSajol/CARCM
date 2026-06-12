#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rice Promoter Classifier V7b - Results Aggregator and Comparison Plotter
=========================================================================
Generates publication-quality comparison plots across all hyperparameter
experiments.

Usage:
    python aggregate_v7b_results.py

@author: saiful
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Publication-quality settings
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

RESULTS_DIR = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/models_v7b/"


def load_all_results():
    """Load all experiment results."""
    all_results = []
    
    for run_name in os.listdir(RESULTS_DIR):
        run_dir = os.path.join(RESULTS_DIR, run_name)
        if not os.path.isdir(run_dir):
            continue
            
        json_file = os.path.join(run_dir, f'{run_name}_results.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                result = {
                    'run_name': run_name,
                    'learning_rate': data['config']['learning_rate'],
                    'hidden_dim': data['config']['hidden_dim'],
                    'dropout': data['config']['dropout'],
                    'batch_size': data['config']['batch_size'],
                    'num_seeds': data['config']['num_seeds'],
                    'test_auc': data['metrics']['test_auc'],
                    'hard_auc': data['metrics']['hard_auc'],
                    'easy_auc': data['metrics']['easy_auc'],
                    'accuracy': data['metrics']['accuracy'],
                    'f1': data['metrics']['f1'],
                    'precision': data['metrics']['precision'],
                    'recall': data['metrics']['recall'],
                    'mcc': data['metrics']['mcc'],
                    'optimal_threshold': data['metrics']['optimal_threshold']
                }
                all_results.append(result)
    
    return pd.DataFrame(all_results)


def plot_hyperparameter_sensitivity(df):
    """Plot hyperparameter sensitivity analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Learning Rate Sensitivity
    ax = axes[0, 0]
    lr_df = df[df['run_name'].str.startswith(('lr_', 'baseline'))].copy()
    lr_df = lr_df.sort_values('learning_rate')
    if len(lr_df) > 0:
        ax.plot(range(len(lr_df)), lr_df['test_auc'], 'o-', markersize=10, label='Test AUC', linewidth=2)
        ax.plot(range(len(lr_df)), lr_df['hard_auc'], 's-', markersize=10, label='Hard AUC', linewidth=2)
        ax.set_xticks(range(len(lr_df)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lr_df['learning_rate']], rotation=45)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('AUC')
        ax.set_title('Learning Rate Sensitivity')
        ax.legend()
        ax.set_ylim([0.75, 1.0])
    
    # Hidden Dimension Sensitivity
    ax = axes[0, 1]
    hd_df = df[df['run_name'].str.startswith(('hd_', 'baseline'))].copy()
    hd_df = hd_df.sort_values('hidden_dim')
    if len(hd_df) > 0:
        ax.plot(hd_df['hidden_dim'], hd_df['test_auc'], 'o-', markersize=10, label='Test AUC', linewidth=2)
        ax.plot(hd_df['hidden_dim'], hd_df['hard_auc'], 's-', markersize=10, label='Hard AUC', linewidth=2)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel('AUC')
        ax.set_title('Hidden Dimension Sensitivity')
        ax.legend()
        ax.set_ylim([0.75, 1.0])
    
    # Dropout Sensitivity
    ax = axes[0, 2]
    do_df = df[df['run_name'].str.startswith(('do_', 'baseline'))].copy()
    do_df = do_df.sort_values('dropout')
    if len(do_df) > 0:
        ax.plot(do_df['dropout'], do_df['test_auc'], 'o-', markersize=10, label='Test AUC', linewidth=2)
        ax.plot(do_df['dropout'], do_df['hard_auc'], 's-', markersize=10, label='Hard AUC', linewidth=2)
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('AUC')
        ax.set_title('Dropout Sensitivity')
        ax.legend()
        ax.set_ylim([0.75, 1.0])
    
    # Batch Size Sensitivity
    ax = axes[1, 0]
    bs_df = df[df['run_name'].str.startswith(('bs_', 'baseline'))].copy()
    bs_df = bs_df.sort_values('batch_size')
    if len(bs_df) > 0:
        ax.plot(bs_df['batch_size'], bs_df['test_auc'], 'o-', markersize=10, label='Test AUC', linewidth=2)
        ax.plot(bs_df['batch_size'], bs_df['hard_auc'], 's-', markersize=10, label='Hard AUC', linewidth=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('AUC')
        ax.set_title('Batch Size Sensitivity')
        ax.legend()
        ax.set_ylim([0.75, 1.0])
    
    # Ensemble Size Sensitivity
    ax = axes[1, 1]
    ns_df = df[df['run_name'].str.startswith(('ns_', 'baseline'))].copy()
    ns_df = ns_df.sort_values('num_seeds')
    if len(ns_df) > 0:
        ax.plot(ns_df['num_seeds'], ns_df['test_auc'], 'o-', markersize=10, label='Test AUC', linewidth=2)
        ax.plot(ns_df['num_seeds'], ns_df['hard_auc'], 's-', markersize=10, label='Hard AUC', linewidth=2)
        ax.set_xlabel('Number of Ensemble Seeds')
        ax.set_ylabel('AUC')
        ax.set_title('Ensemble Size Sensitivity')
        ax.legend()
        ax.set_ylim([0.75, 1.0])
    
    # All configurations comparison
    ax = axes[1, 2]
    df_sorted = df.sort_values('test_auc', ascending=True)
    y_pos = range(len(df_sorted))
    colors = ['green' if 'baseline' in name else 'steelblue' for name in df_sorted['run_name']]
    ax.barh(y_pos, df_sorted['test_auc'], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['run_name'], fontsize=9)
    ax.set_xlabel('Test AUC')
    ax.set_title('All Configurations Ranked')
    ax.axvline(x=df_sorted['test_auc'].max(), color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hyperparameter_sensitivity.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'hyperparameter_sensitivity.pdf'))
    plt.close()
    
    print("  Saved: hyperparameter_sensitivity.png/pdf")


def plot_metrics_comparison(df):
    """Plot comparison of all metrics across experiments."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['test_auc', 'hard_auc', 'accuracy', 'f1']
    titles = ['Test AUC', 'Hard AUC', 'Accuracy', 'F1 Score']
    
    df_sorted = df.sort_values('test_auc', ascending=False)
    x = range(len(df_sorted))
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        colors = ['green' if 'baseline' in name else 'steelblue' 
                  for name in df_sorted['run_name']]
        bars = ax.bar(x, df_sorted[metric], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['run_name'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Configuration')
        
        # Highlight best
        best_idx = df_sorted[metric].argmax()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.pdf'))
    plt.close()
    
    print("  Saved: metrics_comparison.png/pdf")


def plot_heatmap_grid(df):
    """Create heatmap for hyperparameter grid search."""
    
    # Only if we have enough data for a meaningful grid
    unique_lr = df['learning_rate'].unique()
    unique_hd = df['hidden_dim'].unique()
    
    if len(unique_lr) < 2 or len(unique_hd) < 2:
        print("  Not enough data for heatmap grid, skipping...")
        return
    
    # Create pivot table
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LR vs Hidden Dim
    pivot_auc = df.pivot_table(values='test_auc', index='hidden_dim', 
                               columns='learning_rate', aggfunc='mean')
    if pivot_auc.shape[0] > 1 and pivot_auc.shape[1] > 1:
        sns.heatmap(pivot_auc, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0])
        axes[0].set_title('Test AUC: Hidden Dim vs Learning Rate')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Hidden Dimension')
    
    # LR vs Dropout
    pivot_hard = df.pivot_table(values='hard_auc', index='dropout', 
                                columns='learning_rate', aggfunc='mean')
    if pivot_hard.shape[0] > 1 and pivot_hard.shape[1] > 1:
        sns.heatmap(pivot_hard, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Hard AUC: Dropout vs Learning Rate')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Dropout')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hyperparameter_heatmaps.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'hyperparameter_heatmaps.pdf'))
    plt.close()
    
    print("  Saved: hyperparameter_heatmaps.png/pdf")


def plot_radar_comparison(df):
    """Create radar chart comparing top configurations."""
    
    # Select top 5 configurations
    df_top = df.nlargest(5, 'test_auc')
    
    metrics = ['test_auc', 'hard_auc', 'accuracy', 'precision', 'recall', 'f1', 'mcc']
    metric_labels = ['Test AUC', 'Hard AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    
    # Normalize metrics to 0-1 scale for visualization
    normalized = df_top[metrics].copy()
    for col in metrics:
        normalized[col] = (normalized[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_top)))
    
    for idx, (_, row) in enumerate(df_top.iterrows()):
        values = normalized.iloc[idx].values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['run_name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Top 5 Configurations Comparison', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'radar_comparison.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'radar_comparison.pdf'))
    plt.close()
    
    print("  Saved: radar_comparison.png/pdf")


def plot_easy_vs_hard(df):
    """Plot easy vs hard AUC scatter."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(df['easy_auc'], df['hard_auc'], 
                        c=df['test_auc'], cmap='viridis',
                        s=150, alpha=0.8, edgecolors='black')
    
    # Add labels
    for idx, row in df.iterrows():
        ax.annotate(row['run_name'], (row['easy_auc'], row['hard_auc']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Reference line
    ax.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5, label='Easy = Hard')
    
    ax.set_xlabel('Easy Examples AUC')
    ax.set_ylabel('Hard Examples AUC')
    ax.set_title('Easy vs Hard Examples Performance')
    ax.set_xlim([0.7, 1.0])
    ax.set_ylim([0.7, 1.0])
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Test AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'easy_vs_hard.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'easy_vs_hard.pdf'))
    plt.close()
    
    print("  Saved: easy_vs_hard.png/pdf")


def create_latex_table(df):
    """Create LaTeX table for publication."""
    
    df_sorted = df.sort_values('test_auc', ascending=False)
    
    # Select columns for table
    cols = ['run_name', 'learning_rate', 'hidden_dim', 'dropout', 
            'batch_size', 'num_seeds', 'test_auc', 'hard_auc', 'f1']
    
    latex = df_sorted[cols].to_latex(
        index=False,
        float_format="%.4f",
        column_format='l' + 'c' * (len(cols) - 1),
        caption='Hyperparameter search results for rice promoter classification.',
        label='tab:hyperparam_results'
    )
    
    # Save LaTeX table
    with open(os.path.join(RESULTS_DIR, 'results_table.tex'), 'w') as f:
        f.write(latex)
    
    print("  Saved: results_table.tex")


def create_summary_dashboard(df):
    """Create comprehensive summary dashboard."""
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Best configuration metrics
    best = df.loc[df['test_auc'].idxmax()]
    
    # 1. Metrics bar chart
    ax1 = fig.add_subplot(gs[0, 0:2])
    metrics = ['test_auc', 'hard_auc', 'easy_auc', 'accuracy', 'f1', 'precision', 'recall', 'mcc']
    values = [best[m] for m in metrics]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics)))
    bars = ax1.barh(metrics, values, color=colors)
    ax1.set_xlim([0, 1])
    ax1.set_title(f"Best Configuration: {best['run_name']}")
    for bar, val in zip(bars, values):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
    
    # 2. All configs ranked
    ax2 = fig.add_subplot(gs[0, 2:4])
    df_sorted = df.sort_values('test_auc', ascending=True)
    y_pos = range(len(df_sorted))
    colors = ['gold' if name == best['run_name'] else 'steelblue' for name in df_sorted['run_name']]
    ax2.barh(y_pos, df_sorted['test_auc'], color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted['run_name'], fontsize=8)
    ax2.set_xlabel('Test AUC')
    ax2.set_title('All Configurations Ranked')
    
    # 3-6. Hyperparameter sensitivity plots
    params = [
        ('learning_rate', 'lr_', 'Learning Rate'),
        ('hidden_dim', 'hd_', 'Hidden Dimension'),
        ('dropout', 'do_', 'Dropout'),
        ('batch_size', 'bs_', 'Batch Size')
    ]
    
    for idx, (param, prefix, title) in enumerate(params):
        ax = fig.add_subplot(gs[1, idx])
        param_df = df[df['run_name'].str.contains(f'{prefix}|baseline', regex=True)].copy()
        param_df = param_df.sort_values(param)
        if len(param_df) > 0:
            ax.plot(range(len(param_df)), param_df['test_auc'], 'bo-', markersize=8, label='Test')
            ax.plot(range(len(param_df)), param_df['hard_auc'], 'rs-', markersize=8, label='Hard')
            ax.set_xticks(range(len(param_df)))
            if param == 'learning_rate':
                ax.set_xticklabels([f"{x:.0e}" for x in param_df[param]], rotation=45, fontsize=8)
            else:
                ax.set_xticklabels(param_df[param])
            ax.set_xlabel(title)
            ax.set_ylabel('AUC')
            ax.legend(fontsize=8)
            ax.set_ylim([0.75, 1.0])
    
    # 7. Summary statistics
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.axis('off')
    
    summary_text = f"""
    HYPERPARAMETER SEARCH SUMMARY
    =============================
    
    Total Configurations Tested: {len(df)}
    
    Best Configuration: {best['run_name']}
      • Learning Rate: {best['learning_rate']}
      • Hidden Dim: {best['hidden_dim']}
      • Dropout: {best['dropout']}
      • Batch Size: {best['batch_size']}
      • Num Seeds: {best['num_seeds']}
    
    Best Results:
      • Test AUC: {best['test_auc']:.4f}
      • Hard AUC: {best['hard_auc']:.4f}
      • Easy AUC: {best['easy_auc']:.4f}
      • Accuracy: {best['accuracy']:.4f}
      • F1 Score: {best['f1']:.4f}
      • MCC: {best['mcc']:.4f}
    
    Statistics Across Configurations:
      • Test AUC: {df['test_auc'].mean():.4f} ± {df['test_auc'].std():.4f}
      • Hard AUC: {df['hard_auc'].mean():.4f} ± {df['hard_auc'].std():.4f}
    """
    
    ax7.text(0.05, 0.5, summary_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # 8. Distribution of metrics
    ax8 = fig.add_subplot(gs[2, 2:4])
    bp = ax8.boxplot([df['test_auc'], df['hard_auc'], df['easy_auc'], df['f1']],
                     labels=['Test AUC', 'Hard AUC', 'Easy AUC', 'F1'],
                     patch_artist=True)
    colors_box = ['steelblue', 'coral', 'green', 'purple']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax8.set_ylabel('Score')
    ax8.set_title('Metric Distributions Across Configurations')
    
    plt.suptitle('V7b Hyperparameter Search - Summary Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(RESULTS_DIR, 'v7b_summary_dashboard.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'v7b_summary_dashboard.pdf'))
    plt.close()
    
    print("  Saved: v7b_summary_dashboard.png/pdf")


def main():
    print("=" * 60)
    print("V7b Results Aggregator and Comparison Plotter")
    print("=" * 60)
    
    # Load all results
    print("\nLoading all experiment results...")
    df = load_all_results()
    
    if len(df) == 0:
        print("No results found. Please run experiments first.")
        return
    
    print(f"Found {len(df)} experiment results.")
    
    # Save summary CSV
    df_sorted = df.sort_values('test_auc', ascending=False)
    df_sorted.to_csv(os.path.join(RESULTS_DIR, 'all_experiments_summary.csv'), index=False)
    print("\nSaved: all_experiments_summary.csv")
    
    # Print best configuration
    best = df.loc[df['test_auc'].idxmax()]
    print(f"\nBest Configuration: {best['run_name']}")
    print(f"  Test AUC: {best['test_auc']:.4f}")
    print(f"  Hard AUC: {best['hard_auc']:.4f}")
    
    # Generate all plots
    print("\nGenerating comparison plots...")
    
    plot_hyperparameter_sensitivity(df)
    plot_metrics_comparison(df)
    plot_heatmap_grid(df)
    plot_radar_comparison(df)
    plot_easy_vs_hard(df)
    create_latex_table(df)
    create_summary_dashboard(df)
    
    print("\n" + "=" * 60)
    print("All plots saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
