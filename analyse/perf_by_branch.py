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
STRCTURE = 'data/raw/lfs-structure-v3.json'

# =========== Import Custom Modules ===========
from load_data import  OUTPUT_PATH, MODEL_PALETTE
from overall_perf import ensure_directory_exists
from perf_by_lfs import calculate_metrics_by_model_kshot_target

# =========== Auxiliary Functions ===========

def get_subnodes(jn: dict, current_key='root') -> dict:
    mapping = {}
    if isinstance(jn, dict):
        for k, v in jn.items():
            if isinstance(v, dict):
                # If this node has "éléments", add it to mapping
                if "éléments" in v:
                    mapping[k] = list(v["éléments"].keys())
                
                # Recursively explore deeper levels
                sub_mapping = get_subnodes(v, k)
                mapping.update(sub_mapping)
    return mapping

def get_paths_by_terminal_node(structure: dict, current_path: str = "root") -> dict:
    """
    This function recursively traverses the JSON structure and collects paths to each terminal node.
    """
    paths = {}
    if isinstance(structure, dict):
        for k, v in structure.items():
            if v == {}:
                paths[k] = current_path + '.' + k
            else:
                # Create a new path for the recursive call to avoid modifying the current path in place
                new_path = current_path + '.' + k
                # Merge the results of the recursive call into the paths dictionary
                paths.update(get_paths_by_terminal_node(v, current_path=new_path))
    return paths

def dedup_ordered(ls: list):
    """Delete duplicated elements from a list without changing their order."""
    seen = set()
    return [x for x in ls if not (x in seen or seen.add(x))] # Genius

def parse_path(path, node_subnodes):
    return [node for node in path.split('.')[:-1] if ((path.split('.')[-1] not in node_subnodes.keys()) and (node not in {'root', 'éléments'}))]



# =========== plotting functions ===========
def barplot_path(path_nodes: list, node_sub: dict, data: pd.DataFrame, metric: str = "acc_mean"):
    """
    Path is connected by nodes with '.', now we need to get the metrics for every node's subs.
    Specifically, we extract the score of all subnodes for each node in the path as a group of bars,
    by the order from left to the right. Then we put them into the plot, normally we put all nodes'
    subnodes' scores together and merge them by vertical "--" style.
    """

    bars_labels = []
    bars_scores = []
    group_colors = []
    group_means = []
    group_positions = []

    # Colors (fixed by node level)
    colors = [
    '#2176C1',  # 蓝
    '#E07A5F',  # 橙
    '#3A7D44',  # 绿
    '#F2B134',  # 黄
    '#6C4F9C',  # 紫
    '#2A6F77',  # 青
    '#B4557A',  # 粉
    ]

    current_pos = 0  # Record current cumulative y-axis position
    for i, node in enumerate(path_nodes[::-1]):
        subnodes = node_sub.get(node, [])
        group_scores = []
        group_indices = []

        for sub in subnodes:
            match = data[data['target'] == sub]
            if not match.empty:
                bars_labels.append(sub)
                score = match[metric].values[0]
                bars_scores.append(score)
                group_scores.append(score)
                group_colors.append(colors[i % len(colors)])  # Fixed color by node level
                group_indices.append(current_pos)
                current_pos += 1

        if group_scores:
            group_mean = np.mean(group_scores)
            group_means.append((group_mean, group_indices, colors[i % len(colors)]))

        # Insert empty line
        bars_labels.append("")
        bars_scores.append(0)
        group_colors.append((1, 1, 1, 0))  # Transparent color
        current_pos += 1

    # Remove the last empty separator
    if bars_labels and bars_labels[-1] == "":
        bars_labels.pop()
        bars_scores.pop()
        group_colors.pop()

    # Plotting
    sns.set_theme(style="whitegrid", font_scale=1.2)  # Set theme and font scale
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(bars_labels)))  # Adjust figure size
    y_pos = np.arange(len(bars_labels))
    ax.barh(y_pos, bars_scores, color=group_colors)

    # Adding labels
    ax.set_yticks(y_pos)  # Set y-ticks to match the number of bars
    ax.set_yticklabels(bars_labels, fontsize=12)  # Set font size for labels
    ax.set_xlabel(metric, fontsize=14)
    ax.set_title(f"{data['model'].unique()[0]} | Perf branch '{path_nodes[-1]}'", fontsize=13)

    # Adding mean lines for each group
    for mean_val, indices, color in group_means:
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1)
        label_y = max(indices)
        ax.text(mean_val, label_y, f"{mean_val:.2f}", color=color, fontsize=10,
                va='bottom', ha='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

    ax.grid(axis='x', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    ensure_directory_exists(OUTPUT_PATH + 'branchs/')
    plt.savefig(OUTPUT_PATH + 'branchs/'+ f'{data["model"].unique()[0]}_{path_nodes[-1]}.pdf', bbox_inches='tight')
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse performance by branch")
    parser.add_argument('--structure', type=str, default=STRCTURE, help='Path to the JSON structure file')
    args = parser.parse_args()

    # Load the structure
    with open(args.structure, 'r') as f:
        lfs_structure = json.load(f)
    # nodes
    node_subnodes = get_subnodes(lfs_structure)
    print(lfs_structure)
    print(node_subnodes)

    node_mapping_file = "data/processed/lfs-structure-5_nodes_filtrated.json"
    with open(node_mapping_file, 'r') as f:
        lfs_contents=json.load(f)
    filtered_node_subnodes = {}
    for k, v in node_subnodes.items():
    # Check if there's at least one target in v that exists in lfs_contents
        valid_targets = [target for target in v if target in lfs_contents]
        if valid_targets:  # If there are valid targets, keep this node
            filtered_node_subnodes[k] = valid_targets

    # Replace original node_subnodes with filtered version
    node_subnodes = filtered_node_subnodes
    print(node_subnodes)
    # Get paths to terminal nodes
    paths_mapping = get_paths_by_terminal_node(lfs_structure)
    path_list = [tuple(parse_path(path, node_subnodes)) for path in paths_mapping.values()]

    # Get all paths
    path_list = dedup_ordered(path_list)
    # print(path_list)
    path_list = [list(path) for path in path_list if path]  # Remove empty paths


    # Load best performance data
    best_results = pd.read_csv(OUTPUT_PATH + '/best_results.csv')
    perf_best_calculated = calculate_metrics_by_model_kshot_target(best_results)
    print(perf_best_calculated.head())
    # Filtrate scope
    perf_best_calculated = perf_best_calculated[perf_best_calculated['scope_id'] == 1]
    print(perf_best_calculated.shape)
    for model, group in perf_best_calculated.groupby('model'):
        print(f"Model: {model}, Shape: {group.shape}")
        # Iterate through each path
        for path in path_list:
            # Filter the group for the last node
            for mtx in ['acc_mean']:
                barplot_path(path, node_subnodes, group, metric=mtx)