# -*- coding: utf-8 -*-
# ========== MODULE IMPORTS ==========
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# ========== LOCAL IMPORTS ==========
from load_data import *
from overall_perf import *
from perf_by_branch import *
from perf_by_depth import *
from perf_by_lfs import *
from levenshtein import *

# Add the parent directory to the system path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    " ======================== 1. load the data =========================="
    logger.info("Loading data...")
    results = load_multiple_csv(RES)
    results = prepare_data(results)
    logger.info("Data loaded successfully.")

    " ================== 2. Overall performance metrics ==================="
    logger.info("Calculating overall classification report ...")
    evaluate_binary_predictions(results)
    logger.info("Confusion matrix :...")
    plot_confusion_matrices_by_model(results, 
                                     expected_col='expected',
                                     response_col='response',
                                     model_col='model',
                                     labels=['Oui', 'Non'],
                                     label_names=['Yes', 'No'],
                                     savepath=OUTPUT_PATH + 'cnfsn_mtrx_by_models.pdf')
    logger.info("Classification report generated.")
    # Plot confusion matrices by model
    general_perf = calculate_metrics_by_model_kshot(results, output_path=OUTPUT_PATH)
    # plot trends
    logger.info("Plotting trends...")
    plot_metrics_trends(general_perf, output_path=OUTPUT_PATH, metric='acc_mean')
    plot_metrics_trends(general_perf, output_path=OUTPUT_PATH, metric='f1_mean')
    logger.info("General performance metrics calculated.")
    logger.info(f"General performance metrics:\n{general_perf.head()}")
    # Save the general performance metrics
    best_config = extract_best_config(general_perf)
    logger.info(f"Best configuration extracted: {best_config}")
    best_results = filter_best_results(results, best_config)
    logger.info(f"Best results filtered: {best_results.head()}")
    # As well as save the best results
    best_results_path = os.path.join(OUTPUT_PATH, 'best_results.csv')
    best_results.to_csv(best_results_path, index=False)
    logger.info(f"Best results saved to {best_results_path}")

    " ====================== 3. Performance by targets ======================="
    logger.info("Calculating performance by targets...")
    perf_by_target = calculate_metrics_by_model_kshot_target(results)
    logger.info(f"Performance by targets calculated:\n{perf_by_target.head()}")
    perf_by_target.to_csv(OUTPUT_PATH + 'metrics_by_model_target&config.csv', index=False)
    logger.info("Performance by targets saved.")
    # plot performance by targets top N
    logger.info("Plotting performance by targets (top N)...")
    plot_model_perf_by_config_targets(perf_by_target, model_name='Mistral', k_shot=10, kw_context=0, vl_propform=0, 
                                         metric='acc_mean', var_metric='var_by_acc', top_n=15,
                                         output_path=OUTPUT_PATH)
    plot_model_perf_by_config_targets(perf_by_target, model_name='Qwen', k_shot=10, kw_context=0, vl_propform=1,
                                         metric='acc_mean', var_metric='var_by_acc', top_n=15,
                                         output_path=OUTPUT_PATH)
    plot_model_perf_by_config_targets(perf_by_target, model_name='LLaMA', k_shot=10, kw_context=0, vl_propform=1,
                                         metric='acc_mean', var_metric='var_by_acc', top_n=15,
                                         output_path=OUTPUT_PATH)
    logger.info("Performance by targets (top N) plotted.")
    # for best perf
    best_perf_by_target = filter_best_results(perf_by_target, best_config)
    best_perf_by_target.to_csv(OUTPUT_PATH + 'best_perfs.csv', index=False)
    logger.info(f"Now plotting performance by sampled targets for the best configuration")
    groupbarplot_stacked(best_perf_by_target, model_col='model', target_col='target', metric_col='f1_mean', figsize=(16, 12), output_path=OUTPUT_PATH)

    " ====================== 4. Performance by branches ======================="
    # Load the structure
    with open(STRCTURE, 'r') as f:
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
    logger.info(node_subnodes)
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
        logger.info(f"Model: {model}, Shape: {group.shape}")
        # Iterate through each path
        for path in path_list:
            # Filter the group for the last node
            for mtx in ['acc_mean']:
                barplot_path(path, node_subnodes, group, metric=mtx)

    " ====================== 5. Performance by depth ======================="
    logger.info("Calculating performance by depth...")
    with open('data/raw/lfs-structure-v3-simplified.json', 'r', encoding='utf-8') as f:
        lfs_structure = json.load(f)
    # Load performance data
    best_perf = pd.read_csv(OUTPUT_PATH + 'best_perfs.csv')
    # 筛选scope_id为1的targets
    best_perf = best_perf[best_perf['scope_id'] == 1].copy()
    # 直接画图
    plot_perf_by_depth(best_perf, scope_col='scope_id', target_col='target', lfs_structure=lfs_structure, scope=1, save=True, output_path=OUTPUT_PATH, show=False)

    " =========================== 6. Levenshtein ============================"
    questions_file = 'experiments/task_binary_3/data/questions-2.json'
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    # 提取paradigmatic和syntagmatic的target
    targets = list(questions.keys())
    stop = 'FL_syntagmatiques'
    i = targets.index(stop)
    TGT_PARA = targets[:i]
    # print(f"Paradigmatic targets: {TGT_PARA}")
    TGT_SYNT = targets[i:]
    # print(f"Syntagmatic targets: {TGT_SYNT}")
    # ========== 画图的目录 ==========
    results_best = pd.read_csv(OUTPUT_PATH + 'best_results.csv')
    # 筛选scope
    results_best = results_best[results_best['scope_id'] == 1].copy()
    # Add Levenshtein ratio
    results_best['quest_levratio'] = results_best.apply(lambda row: levenshtein_ratio(row['q_kw'], row['q_vl']), axis=1)
    ensure_directory_exists(OUTPUT_PATH + 'levenshtein/')

    # 计算 Levenshtein 相似度与 polarity 的相关性
    lev_corr_df = corl_levratio_polarity(results_best, lev_col='quest_levratio', target_col='target', model_name=None, perspective='pred')

    # 1. 三个模型都来着, 简单图
    with sns.axes_style("whitegrid"):
        barchart_lr_plrty(results_best, lev_col='quest_levratio', target_col='target', p_filtred=True, model_name='Qwen')
        barchart_lr_plrty(results_best, lev_col='quest_levratio', target_col='target', p_filtred=True, model_name='LLaMA')
        barchart_lr_plrty(results_best, lev_col='quest_levratio', target_col='target', p_filtred=True, model_name='Mistral')

    # 2. 画histogram
    histogram_lr_plrty(results_best, lev_col='quest_levratio', p_filtred=True, target_col='target', para_synt=True, para_tgt=TGT_PARA, synt_tgt=TGT_SYNT)

    # 3. 画boxplot
    boxplot_lr_plrty(results_best, lev_col='quest_levratio', p_filtred=True, target_col='target', para_synt=True, para_tgt=TGT_PARA, synt_tgt=TGT_SYNT)

    # 4. 全图
    barchart_lr_plrty_pred_truth(
        results_best, 
        lev_col='quest_levratio', 
        target_col='target', 
        p_filtred=True, 
        model_name=None,
        Modelpalette=MODEL_PALETTE
    )
    # 5. subset targets
    barchart_lr_plrty_pred_truth(
    data=results_best,
    lev_col='quest_levratio',
    target_col='target',
    targets_to_plot= ['V0',  'A0', 'Contr', 'Pred'],
    Modelpalette= MODEL_PALETTE,
    p_filtred=False,
    title="Correlation for Key Targets"
)
    # 6. 计算正负例差值与 polarity 相关性
    questions_instance = pd.read_csv('data/interim/instances_extracted.csv')

    diff_df = calculate_positive_negative_difference(questions_instance, lev_col='lev_ratio', usage_col='usage', qid_col='Q_id', pos_label='prmpt_pos', neg_label='prmpt_neg')
    diff_merged = diff_df.merge(
        lev_corr_df,
        on = 'target',
        how = 'left'
    )
    print(diff_merged.head())

    scatter_diff_vs_polarity(diff_merged, diff_col='difference', polarity_col='correlation_pred')

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis...")
    try:
        main()
        logger.info("Analysis completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        sys.exit(1)





