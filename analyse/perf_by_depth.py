import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ========= add ROOT_DIR to sys.path ==========
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# =========== Constants ===========

# =========== Import Custom Modules ===========
from load_data import OUTPUT_PATH, MODEL_PALETTE
from overall_perf import ensure_directory_exists


# =========== Auxiliary Functions ===========
def get_paths(tree, prefix=None, skip_keys={"FL_paradigmatiques", "FL_syntagmatiques"}, path_map=None):
    if prefix is None:
        prefix = []
    if path_map is None:
        path_map = {}

    for key, subtree in tree.items():
        # 构造路径，并跳过顶级标识
        next_prefix = prefix + ([] if key in skip_keys else [key])

        # 只记录一次（首次出现的路径）
        if key not in skip_keys and key not in path_map:
            path_map[key] = next_prefix

        # 继续递归
        if isinstance(subtree, dict):
            get_paths(subtree, next_prefix, skip_keys, path_map)

    return path_map


def plot_perf_by_depth(
    df: pd.DataFrame,
    lfs_structure: dict,
    metrics: list = None,
    scope_col: str = 'scope_id',
    target_col: str = 'target',
    scope: int = 1,
    output_path: str = OUTPUT_PATH,
    show: bool = True,
    save: bool = False,
    figsize: tuple = None,
    legend_loc: str = 'best'
):
    """
    按深度分组画模型性能趋势。
    metrics: None 或 ['acc_mean'] 只画accuracy，或 ['acc_mean', ...] 画多指标。
    output_path: 保存路径（含文件名），如 None 不保存。
    show: 是否 plt.show()
    save: 是否保存图片
    figsize: 画布大小，None时自动选择
    """
    node_paths = get_paths(lfs_structure)
    df = df.copy()
    df['depth'] = df[target_col].map(lambda x: len(node_paths.get(x, [])) if x in node_paths else 0)
    if scope is not None:
        df_filtered = df[df[scope_col] == scope]
    else:
        df_filtered = df

    if metrics is None:
        metrics = ['acc_mean']
    if len(metrics) == 1:
        # 单图
        metric = metrics[0]
        if figsize is None:
            figsize = (6.8, 5.2)
        sns.set_context("paper", font_scale=1.5)
        plt.figure(figsize=figsize)
        for model in df_filtered['model'].unique():
            model_data = df_filtered[df_filtered['model'] == model]
            sns.lineplot(
                data=model_data,
                x='depth',
                y=metric,
                label=model,
                linestyle='solid',
                linewidth=2,
                marker='o',
                markersize=5,
                color=MODEL_PALETTE[model],
                errorbar=None
            )
        plt.xlabel("Depth")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(sorted(df_filtered['depth'].unique()))
        plt.tick_params(axis='both', labelsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Model', fontsize=12, title_fontsize=14, frameon=False, loc=legend_loc)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save and output_path:
            plt.savefig(output_path + 'perf_by_depth_acc.pdf', format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    else:
        # 多指标子图
        if figsize is None:
            figsize = (16, 12)
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
        axes = axes.flatten()
        sns.set_context("talk")
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.lineplot(
                data=df_filtered,
                x='depth',
                y=metric,
                hue='model',
                marker='o',
                palette=MODEL_PALETTE,
                errorbar=None,
                ax=ax
            )
            ax.set_xlabel("Depth")
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_xticks(range(df_filtered['depth'].min(), df_filtered['depth'].max() + 1))
            ax.tick_params(axis='x', labelsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title='Model', loc=legend_loc)
        plt.tight_layout()
        if save and output_path:
            plt.savefig(output_path + 'perf_by_depth_multiscore.pdf', format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

if __name__ == "__main__":
    # Load the structure JSON file
    with open('data/raw/lfs-structure-v3-simplified.json', 'r', encoding='utf-8') as f:
        lfs_structure = json.load(f)
    # Load performance data
    best_perf = pd.read_csv(OUTPUT_PATH + 'best_perfs.csv')
    # 筛选scope_id为1的targets
    best_perf = best_perf[best_perf['scope_id'] <= 2].copy()
    # 直接画图
    plot_perf_by_depth(best_perf, scope_col='scope_id', target_col='target', lfs_structure=lfs_structure, scope=None, save=True, output_path=OUTPUT_PATH, show=False)
