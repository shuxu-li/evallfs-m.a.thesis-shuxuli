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
import Levenshtein
from scipy.stats import pearsonr
# ========= add ROOT_DIR to sys.path ==========
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

# =========== Import Custom Modules ===========
from load_data import OUTPUT_PATH, MODEL_PALETTE
from overall_perf import ensure_directory_exists
# =========== Constants ===========

# =========== Auxiliary Functions ===========
def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    return Levenshtein.distance(s1, s2)
def levenshtein_ratio(s1, s2):
    """
    Calculate the Levenshtein ratio between two strings.
    """
    return Levenshtein.ratio(s1, s2)

def corl_levratio_polarity(data: pd.DataFrame, lev_col: str = 'quest_levratio', target_col='target', model_name='Qwen', perspective='pred'):
    """
    以模型分组，以target为单位计算 Levenshtein 相似度与 polarity 的相关性。
    如果perspective='p'，就是看response（Oui/Non）与 Levenshtein 相似度的相关性；
    如果perspective='t'，就是看expected （Oui/Non）与 Levenshtein 相似度的相关性。
    """
    if model_name:
        data = data[data['model'] == model_name].copy()
    if perspective == 'true':
        data['polarity'] = data['expected'].copy().map({'Oui': 1, 'Non': -1})
    if perspective == 'pred':
        data['polarity'] = data['response'].copy().map({'Oui': 1, 'Non': -1})
    results = []
    for model, group in data.groupby('model'):
        for target, sub_group in group.groupby(target_col):
            if sub_group.empty:
                continue
            # 计算相关系数
            x = sub_group[lev_col]
            y = sub_group['polarity']

            if x.nunique() < 2 or y.nunique() < 2:
                corr, p_value = np.nan, np.nan
            else:
                corr, p_value = pearsonr(x, y)
            # corr, p_value = pearsonr(sub_group[lev_col], sub_group['polarity'])
            results.append({
                'model': model,
                'target': target,
                f'correlation_{perspective}': round(corr,3),
                f'p_value_{perspective}': round(p_value,3)
            })
    return pd.DataFrame(results)

def barchart_lr_plrty(data: pd.DataFrame, lev_col: str = 'quest_levratio', target_col='target', p_filtred: bool = True, model_name: str = None, synt_para: bool = False):
   
    # Filter data for the specified model if model_name is provided
    if model_name:
        data = data[data['model'] == model_name].copy()
    
    # Calculate correlation
    df_cor = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=model_name, perspective='pred')
    
    if p_filtred:
        # Filter by p_value < 0.05
        df_cor = df_cor[df_cor['p_value_pred'] < 0.05]
        
        # Retain only targets common to all three models
        common_targets = set(data['target'].unique())
        for model in data['model'].unique():
            common_targets &= set(data[data['model'] == model]['target'].unique())
        
        # Filter the correlation DataFrame to include only common targets
        df_cor = df_cor[df_cor['target'].isin(common_targets)]
    
    # Sort by correlation
    df_cor = df_cor.sort_values('correlation_pred', ascending=False)
    
    # Plot histogram
    plt.figure(figsize=(14, 8))
    sns.set_context("talk")
    sns.histplot(
        data=df_cor,
        x='target',
        hue='model',
        weights='correlation_pred',
        multiple='dodge',  # Arrange bars side by side
        palette=MODEL_PALETTE,
        binwidth=0.1,  # Set narrower bars
    )
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('Target')
    plt.ylabel('Correlation with Polarity')
    plt.title(f'Levenshtein Similarity vs Polarity Correlation Histogram\n{"Filtered" if p_filtred else "All"}')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + f'levenshtein/correl_levratio_polarity_{model_name}.pdf', format='pdf')
    plt.show()

# 把整个的target的相关性分布画出来，histogram展示
def histogram_lr_plrty(data: pd.DataFrame, lev_col: str = 'quest_levratio', target_col='target', p_filtred: bool = True, para_synt: bool = True, para_tgt: list = None, synt_tgt: list = None):
    """
    如果para_synt=True,先拆paradigmatic和syntagmatic两种类型的target，然后分成上下两个子图；
    否则就画一个图。
    使用sns.histplot绘制相关系数分段下的目标数量直方图，支持所有模型。
    """
    data = data.copy()  # 避免 SettingWithCopyWarning
    data['polarity'] = data['response'].map({'Oui': 1, 'Non': 0})

    if para_synt:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for i, (label, targets) in enumerate({'Paradigmatic': para_tgt, 'Syntagmatic': synt_tgt}.items()):
            subset = data[data[target_col].isin(targets)].copy()
            corr_df = corl_levratio_polarity(subset, lev_col=lev_col, target_col=target_col, model_name=None)
            print(corr_df.columns)
            if p_filtred:
                corr_df = corr_df[corr_df['p_value_pred'] < 0.05]
            
            sns.histplot(
                data=corr_df,
                x='correlation_pred',
                hue='model',
                bins=np.arange(-1, 1.1, 0.2),
                kde=True,
                palette=MODEL_PALETTE,
                alpha=0.3,
                ax=axes[i]
            )
            axes[i].set_title(f"{label} Targets")
            axes[i].set_ylabel("Target Count")
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel("Correlation")
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'levenshtein/correl_levratio_polarity__histo.pdf', format='pdf')
        plt.show()

    else:
        # 整体数据合并画一个图
        corr_df = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=None)
        if p_filtred:
            corr_df = corr_df[corr_df['p_value_pred'] < 0.05]

        plt.figure(figsize=(12, 8))
        sns.histplot(
            data=corr_df,
            x='correlation_pred',
            hue='model',
            bins=np.arange(-1, 1.1, 0.2),
            kde=True,
            palette=MODEL_PALETTE,
            alpha=0.3
        )
        plt.title("Target Count by Correlation Bins (All Models)")
        plt.xlabel("Correlation")
        plt.ylabel("Target Count")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'levenshtein/correl_levratio_polarity__histo.pdf', format='pdf')
        plt.show()

# 从histogram转换成boxplot
def boxplot_lr_plrty(data: pd.DataFrame, lev_col: str = 'quest_levratio', target_col='target', p_filtred: bool = True, para_synt: bool = True, para_tgt: list = None, synt_tgt: list = None):
    """
    如果para_synt=True, 先拆paradigmatic和syntagmatic两种类型的target，分成上下两个子图；
    否则画一个图。
    使用sns.boxplot绘制不同模型下的correlation分布，按model分颜色。
    """
    data = data.copy()
    data['polarity'] = data['response'].map({'Oui': 1, 'Non': 0})

    if para_synt:
        fig, axes = plt.subplots(1, 2, figsize=(8,6), sharex=True)
        y_min, y_max = None, None  # 用于记录y轴范围
        for i, (label, targets) in enumerate({'Paradigmatic': para_tgt, 'Syntagmatic': synt_tgt}.items()):
            subset = data[data[target_col].isin(targets)].copy()
            corr_df = corl_levratio_polarity(subset, lev_col=lev_col, target_col=target_col, model_name=None)
            if p_filtred:
                corr_df = corr_df[corr_df['p_value_pred'] < 0.05]

            sns.boxplot(
                data=corr_df,
                x='model',
                y='correlation_pred',
                hue='model',
                palette=MODEL_PALETTE,
                legend=False,
                ax=axes[i]
            )
            axes[i].set_title(f"{label} Targets")
            axes[i].set_ylabel("Pearson r (similarity vs. polarity)")
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)

            # 更新y轴范围
            current_y_min, current_y_max = axes[i].get_ylim()
            y_min = min(y_min, current_y_min) if y_min is not None else current_y_min
            y_max = max(y_max, current_y_max) if y_max is not None else current_y_max
         # 设置统一的y轴范围
        for ax in axes:
            ax.set_ylim(y_min, y_max)

        plt.supxlabel = "Model"
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'levenshtein/correl_levratio_polarity_boxplot.pdf', format='pdf')
        plt.show()

    else:
        corr_df = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=None)
        if p_filtred:
            corr_df = corr_df[corr_df['p_value'] < 0.05]

        plt.figure(figsize=(6,6))
        sns.boxplot(
            data=corr_df,
            x='model',
            y='correlation_pred',
            hue='model',
            palette=MODEL_PALETTE,
            legend=False
        )
        plt.title("Correlation Distribution by Model (p < 0.05)")
        plt.xlabel("Model")
        plt.ylabel("Pearson r (similarity vs. polarity)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'levenshtein/correl_levratio_polarity_boxplot.pdf', format='pdf')
        plt.show()

# # 可以画一个考虑背景（真实相关性的情况）
# def barchart_lr_plrty_pred_truth(
#     data: pd.DataFrame, 
#     lev_col: str = 'quest_levratio', 
#     target_col: str = 'target', 
#     model_name: str = None,
#     Modelpalette: dict = None,
#     targets_to_plot: list = None,   # 指定 subset targets
#     p_filtred: bool = True,         # 是否过滤 p 值
#     synt_para: bool = True,  # 是否分paradigmatic和syntagmatic
#     model_order: list = ['Qwen', 'LLaMA', 'Mistral'],
#     title: str = None,
#     savepath: str = OUTPUT_PATH
# ):
#     """
#     绘制目标 target 的 polarity correlation。
#     支持全量和 subset 模式；支持 p 值过滤。
#     """
#     data = data[data['scope_id'] == 1].copy()

#     all_results_pred = []
#     all_results_true = []

#     for m in data['model'].unique():
#         df_cor_pred = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=m, perspective='pred')
#         df_cor_true = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=m, perspective='true')
#         df_cor_pred['model'] = m
#         df_cor_true['model'] = m
#         all_results_pred.append(df_cor_pred)
#         all_results_true.append(df_cor_true)

#     df_cor_pred = pd.concat(all_results_pred, ignore_index=True)
#     df_cor_true = pd.concat(all_results_true, ignore_index=True)

#     if targets_to_plot:
#         df_cor_pred = df_cor_pred[df_cor_pred['target'].isin(targets_to_plot)]
#         df_cor_true = df_cor_true[df_cor_true['target'].isin(targets_to_plot)]

#     if p_filtred:
#         df_cor_true = df_cor_true[df_cor_true['p_value_true'] < 0.05]
#         df_cor_pred = df_cor_pred[df_cor_pred['target'].isin(df_cor_true['target'])]

#     df_cor_true = df_cor_true.sort_values('correlation_true', ascending=False)
#     unique_targets = df_cor_true['target'].unique()
#     df_cor_true['target'] = pd.Categorical(df_cor_true['target'], categories=unique_targets, ordered=True)
#     df_cor_pred = pd.merge(df_cor_pred, df_cor_true[['target', 'model', 'correlation_true']], on=['target', 'model'], how='inner')
#     df_cor_pred['target'] = pd.Categorical(df_cor_pred['target'], categories=unique_targets, ordered=True)

#     # Plot
#     num_targets = len(unique_targets)
#     fig_width = max(8, 0.5 * num_targets)  # 横轴更宽
#     fig_height = min(0.4 * num_targets + 2, 20)  # 控最大高度为20

#     plt.figure(figsize=(fig_width, fig_height))
#     # plt.figure(figsize=(6.8, 0.3 * len(unique_targets) + 3))
#     sns.set_theme(style="whitegrid")
#     sns.set_context("paper", font_scale=1.3)

#     sns.barplot(
#         data=df_cor_true,
#         x='target',
#         y='correlation_true',
#         color='lightcoral',
#         alpha=0.5,
#         width=0.7,
#         dodge=False,
#         edgecolor='black',
#         label='True Correlation',
#         errorbar=None
#     )

#     df_cor_pred['model'] = pd.Categorical(df_cor_pred['model'], categories=model_order, ordered=True)
#     sns.barplot(
#         data=df_cor_pred,
#         x='target',
#         y='correlation_pred',
#         hue='model',
#         palette=Modelpalette,
#         width=0.7,
#         alpha=0.9,
#         dodge=True,
#         edgecolor='black',
#         errorbar=None
#     )

#     plt.xticks(rotation=90, ha='right')
#     plt.xlabel('')
#     plt.ylabel('Correlation with Polarity')
#     if title:
#         plt.title(title, fontsize=16)
#     plt.legend(title='Model Prediction Correlation', fontsize=11, title_fontsize=12)
#     plt.grid(axis='both', linestyle='--', alpha=0.9)
#     plt.tight_layout()
#     plt.savefig(savepath + f'levenshtein/correl_true_pred_{title}.pdf', format='pdf', bbox_inches='tight')
#     plt.show()

def barchart_lr_plrty_pred_truth(
    data: pd.DataFrame, 
    lev_col: str = 'quest_levratio', 
    target_col: str = 'target', 
    model_name: str = None,
    Modelpalette: dict = None,
    targets_to_plot: list = None,
    p_filtred: bool = True,
    synt_para: bool = True,
    model_order: list = ['Qwen', 'LLaMA', 'Mistral'],
    title: str = None,
    savepath: str = OUTPUT_PATH
):
    """
    绘制目标 target 的 polarity correlation。
    支持全量和 subset 模式；支持 p 值过滤；支持 syntagmatic vs paradigmatic 分组绘图。
    """
    data = data[data['scope_id'] == 1].copy()

    all_results_pred = []
    all_results_true = []

    for m in data['model'].unique():
        df_cor_pred = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=m, perspective='pred')
        df_cor_true = corl_levratio_polarity(data, lev_col=lev_col, target_col=target_col, model_name=m, perspective='true')
        df_cor_pred['model'] = m
        df_cor_true['model'] = m
        all_results_pred.append(df_cor_pred)
        all_results_true.append(df_cor_true)

    df_cor_pred = pd.concat(all_results_pred, ignore_index=True)
    df_cor_true = pd.concat(all_results_true, ignore_index=True)

    if targets_to_plot:
        df_cor_pred = df_cor_pred[df_cor_pred['target'].isin(targets_to_plot)]
        df_cor_true = df_cor_true[df_cor_true['target'].isin(targets_to_plot)]

    if p_filtred:
        df_cor_true = df_cor_true[df_cor_true['p_value_true'] < 0.05]
        df_cor_pred = df_cor_pred[df_cor_pred['target'].isin(df_cor_true['target'])]

    # ---------------- 分 synt / para 两张图 ---------------- #
    if synt_para:
        # 你需要自己提供这两个列表：
        synt = TGT_SYNT
        para = TGT_PARA

        groups = {
            'Syntagmatic': synt,
            'Paradigmatic':  para
        }

        for group_name, target_list in groups.items():
            df_true_sub = df_cor_true[df_cor_true['target'].isin(target_list)].copy()
            df_pred_sub = df_cor_pred[df_cor_pred['target'].isin(target_list)].copy()

            if df_true_sub.empty or df_pred_sub.empty:
                print(f"Skipping {group_name} group due to empty data.")
                continue

            df_true_sub = df_true_sub.sort_values('correlation_true', ascending=False)
            unique_targets = df_true_sub['target'].unique()
            df_true_sub['target'] = pd.Categorical(df_true_sub['target'], categories=unique_targets, ordered=True)
            df_pred_sub = pd.merge(df_pred_sub, df_true_sub[['target', 'model', 'correlation_true']], on=['target', 'model'], how='inner')
            df_pred_sub['target'] = pd.Categorical(df_pred_sub['target'], categories=unique_targets, ordered=True)

            # 绘图
            num_targets = len(unique_targets)
            fig_width = max(8, 0.5 * num_targets)
            # fig_height = min(0.4 * num_targets + 2, 20)
            fig_height = 6

            plt.figure(figsize=(fig_width, fig_height))
            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.4)

            sns.barplot(
                data=df_true_sub,
                x='target',
                y='correlation_true',
                color='lightcoral',
                alpha=0.5,
                width=0.7,
                dodge=False,
                edgecolor='black',
                label='True Correlation',
                errorbar=None
            )

            df_pred_sub['model'] = pd.Categorical(df_pred_sub['model'], categories=model_order, ordered=True)
            sns.barplot(
                data=df_pred_sub,
                x='target',
                y='correlation_pred',
                hue='model',
                palette=Modelpalette,
                width=0.7,
                alpha=0.9,
                dodge=True,
                edgecolor='black',
                errorbar=None
            )

            plt.xticks(rotation=90, ha='right')
            plt.xlabel('')
            plt.ylabel('Correlation with Polarity')
            plt.title(f"{group_name} targets", fontsize=16)
            plt.legend(title='Model Prediction Correlation', fontsize=11, title_fontsize=12)
            plt.grid(axis='both', linestyle='--', alpha=0.9)
            plt.tight_layout()
            plt.savefig(savepath + f'levenshtein/correl_true_pred_{title}_{group_name}.pdf', format='pdf', bbox_inches='tight')
            plt.show()

    # ---------------- 原版：单张图 ---------------- #
    else:
        df_cor_true = df_cor_true.sort_values('correlation_true', ascending=False)
        unique_targets = df_cor_true['target'].unique()
        df_cor_true['target'] = pd.Categorical(df_cor_true['target'], categories=unique_targets, ordered=True)
        df_cor_pred = pd.merge(df_cor_pred, df_cor_true[['target', 'model', 'correlation_true']], on=['target', 'model'], how='inner')
        df_cor_pred['target'] = pd.Categorical(df_cor_pred['target'], categories=unique_targets, ordered=True)

        num_targets = len(unique_targets)
        fig_width = max(8, 0.5 * num_targets)
        # fig_height = min(0.4 * num_targets + 2, 20)
        fig_height = 6

        plt.figure(figsize=(fig_width, fig_height))
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.3)

        sns.barplot(
            data=df_cor_true,
            x='target',
            y='correlation_true',
            color='lightcoral',
            alpha=0.5,
            width=0.7,
            dodge=False,
            edgecolor='black',
            label='True Correlation',
            errorbar=None
        )

        df_cor_pred['model'] = pd.Categorical(df_cor_pred['model'], categories=model_order, ordered=True)
        sns.barplot(
            data=df_cor_pred,
            x='target',
            y='correlation_pred',
            hue='model',
            palette=Modelpalette,
            width=0.7,
            alpha=0.9,
            dodge=True,
            edgecolor='black',
            errorbar=None
        )

        plt.xticks(rotation=90, ha='right')
        plt.xlabel('')
        plt.ylabel('Correlation with Polarity')
        if title:
            plt.title(title, fontsize=16)
        plt.legend(title='Model Prediction Correlation', fontsize=11, title_fontsize=12)
        plt.grid(axis='both', linestyle='--', alpha=0.9)
        plt.tight_layout()
        plt.savefig(savepath + f'levenshtein/correl_true_pred_{title}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
# ============ levratio_difference vs polarity ============
def calculate_positive_negative_difference(df: pd.DataFrame, lev_col: str = 'lev_ratio', usage_col: str = 'usage', qid_col='Q_id', pos_label: str = 'prmpt_pos', neg_label: str = 'prmpt_neg'):
    """
    按照target分组计算每target所有问题的正负例的 Levenshtein 相似度差值的平均。
    只看scope_id == 1 的数据。
    只比较prompt的正负例，不考虑问题类型。
    返回一个 DataFrame，包含问题、正例平均值、负例平均值和差值。

    """
    df_filtered = df[df['scope_id'] == 1]
    results = []

    for target, group in df_filtered.groupby('target'):
        pos_vals = group[group[usage_col] == pos_label][lev_col].values
        neg_vals = group[group[usage_col] == neg_label][lev_col].values

        # 如果数据不足，则跳过
        if len(pos_vals) < 2 or len(neg_vals) < 2:
            continue

        # 计算平均值
        pos_mean = np.mean(pos_vals)
        neg_mean = np.mean(neg_vals)
        diff = pos_mean - neg_mean

        results.append({
            'target': target,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'difference': diff
        })
    return pd.DataFrame(results)

def scatter_diff_vs_polarity(df: pd.DataFrame, diff_col: str = 'difference', polarity_col: str = 'correlation'):
    """
    绘制正负例差值与 polarity 相关性的散点图，并分模型添加回归曲线。
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(6.8, 5.2))
    
    # 绘制散点图
    sns.set_context("paper", font_scale=1.4)
    sns.scatterplot(data=df, x=diff_col, y=polarity_col, hue='model', style='model', palette=MODEL_PALETTE, s=100)
    
    # 分模型添加回归曲线
    for model in df['model'].dropna().unique():  # 👈 跳过 NaN
        if model not in MODEL_PALETTE:
            continue  # 👈 避免未知 model 报错
        model_data = df[df['model'] == model]
        sns.regplot(
            data=model_data,
            x=diff_col,
            y=polarity_col,
            scatter=False,
            label=f"{model} Regression",
            line_kws={"lw": 2},
            color=MODEL_PALETTE[model]
        )
    
    # plt.title("Positive-Negative Difference vs Polarity (By Model)")
    plt.xlabel(diff_col.replace('_', ' ').title())
    plt.ylabel(polarity_col.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + 'levenshtein/diff_vs_polarity.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()


# ============ Main Function ============
if __name__ == "__main__":
    # 分出paradigmatic和syntagmatic两种类型的target
    questions_file = 'experiments/task_binary_3/data/questions.json'
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
        synt_para=True,
        model_name=None,
        Modelpalette=MODEL_PALETTE
    )
    # 5. subset targets
    barchart_lr_plrty_pred_truth(
    data=results_best,
    lev_col='quest_levratio',
    target_col='target',
    targets_to_plot= ['Able_i', 'V0', 'S0', 'Syn', 'S1', 'A0', 'Gener', 'A_dériv_ajout_de_sens', 'N_dériv_ajout_de_sens', 'Pred', 'Qual1', 'Nom_gouverneur', 'Colloc_préposition', 'Verbes_de_phase' ],
    synt_para=False,
    Modelpalette= MODEL_PALETTE,
    p_filtred=True,
    title="Correlation similarity-polarity for FL selected"
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
