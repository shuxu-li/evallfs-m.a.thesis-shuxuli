import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from load_data import load_multiple_csv, prepare_data, RES, MODEL_PALETTE, SUBSAMPLE_PARA, SUBSAMPLE_SYNTAGM, OUTPUT_PATH
from overall_perf import ensure_directory_exists, extract_best_config, filter_best_results, calculate_metrics_by_model_kshot

# add ROOT_DIR to sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))
# print(f"ROOT_DIR: {ROOT_DIR}")
# ============== Utils ===============
def calculate_metrics_by_model_kshot_target(df, true_col='expected', pred_col='response', model_col='model', kshot_col='k-shot', target_col='target', repeat_col='repeat_num',context_col='kw_context', propform_col='vl_propform', scope_col='scope_id', decimal_places=3):
    
    results_list = []

    for (model, kshot, target, scope, kw_context, vl_propform), group in df.groupby([model_col, kshot_col, target_col, scope_col, context_col, propform_col]):
        metrics = {
            'model': model,
            'k-shot': kshot,
            'target': target,
            'scope_id': scope,
            'kw_context': kw_context,
            'vl_propform': vl_propform
        }
        
        acc_values, prec_values, rec_values, f1_values = [], [], [], []
        
        for repeat in sorted(group[repeat_col].unique()):
            repeat_group = group[group[repeat_col] == repeat]
            
            if len(repeat_group) < 2 or repeat_group[true_col].nunique() < 2:
                continue
                
            acc = accuracy_score(repeat_group[true_col], repeat_group[pred_col])
            prec = precision_score(repeat_group[true_col], repeat_group[pred_col], average='weighted', zero_division=0)
            rec = recall_score(repeat_group[true_col], repeat_group[pred_col], average='weighted', zero_division=0)
            f1 = f1_score(repeat_group[true_col], repeat_group[pred_col], average='weighted', zero_division=0)
            
            acc_values.append(acc)
            prec_values.append(prec)
            rec_values.append(rec)
            f1_values.append(f1)
        
        if not acc_values:
            continue
        
        # aggregate metrics across repeats
        metrics.update({
            'acc_mean': round(np.mean(acc_values), decimal_places),
            'prec_mean': round(np.mean(prec_values), decimal_places),
            'rec_mean': round(np.mean(rec_values), decimal_places),
            'f1_mean': round(np.mean(f1_values), decimal_places),
            'var_by_acc': round(np.var(acc_values), 6),
            'var_by_f1': round(np.var(f1_values), 6)
        })
        
        results_list.append(metrics)
    
    results_df = pd.DataFrame(results_list)

    column_order = ['model', 'k-shot', 'target', 'scope_id', 'kw_context', 'vl_propform', 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean', 'var_by_acc', 'var_by_f1']
    results_df = results_df[column_order]
    
    return results_df.sort_values(['model', 'k-shot', 'target', 'acc_mean'], ascending=[True, True, True, False])

def plot_model_perf_by_config_targets(df, model_name, k_shot=None, kw_context=None, vl_propform=None,
                                         metric='f1_mean', var_metric='var_by_f1', top_n=15, figsize=(20, 8),
                                         output_path=None):
    """
    Seaborn version of performance bar plot for a model under specific configuration.
    Shows top and bottom N targets.
    """
    assert metric in ['f1_mean', 'acc_mean'], "Invalid metric. Choose 'f1_mean' or 'acc_mean'."
    assert var_metric in ['var_by_f1', 'var_by_acc'], "Invalid variance metric."

    # Filter data
    filtered_df = df[df['model'] == model_name].copy()
    if k_shot is not None:
        filtered_df = filtered_df[filtered_df['k-shot'] == k_shot]
    if kw_context is not None:
        filtered_df = filtered_df[filtered_df['kw_context'] == kw_context]
    if vl_propform is not None:
        filtered_df = filtered_df[filtered_df['vl_propform'] == vl_propform]

    config_str = f"model={model_name}"
    if k_shot is not None:
        config_str += f", k-shot={k_shot}"
    if kw_context is not None:
        config_str += f", kw_ctx={kw_context}"
    if vl_propform is not None:
        config_str += f", vl_pf={vl_propform}"

    agg_df = filtered_df.groupby('target')[[metric, var_metric]].mean().reset_index()

    if len(agg_df) == 0:
        print(f"No data for configuration: {config_str}")
        return

    available_targets = len(agg_df)
    if available_targets < 2 * top_n:
        top_n = available_targets // 2 or 1
        print(f"Adjusted top_n to {top_n} due to limited targets ({available_targets} available)")

    top_targets = agg_df.nlargest(top_n, metric)
    bottom_targets = agg_df.nsmallest(top_n, metric).sort_values(metric, ascending=False)

    top_targets['group'] = 'TOP'
    bottom_targets['group'] = 'BOTTOM'
    combined = pd.concat([top_targets, bottom_targets], axis=0)

    # To preserve order
    combined['target'] = pd.Categorical(combined['target'], categories=combined['target'], ordered=True)

    # Plot
    plt.figure(figsize=figsize)
    sns.set(style="whitegrid", context="notebook")

    ax = sns.barplot(
        data=combined,
        x='target',
        y=metric,
        hue='group',
        dodge=False,
        palette={'TOP': 'tab:blue', 'BOTTOM': 'tab:orange'},
        errorbar=None
    )

    # Add error bars manually
    yerr = np.sqrt(combined[var_metric])
    yerr = np.minimum(yerr, combined[metric] * 0.5)
    for i, (bar, err) in enumerate(zip(ax.patches, yerr)):
        height = bar.get_height()
        ax.errorbar(bar.get_x() + bar.get_width() / 2, height, yerr=err, fmt='none', ecolor='gray', capsize=4)

    # Value labels
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.03, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, rotation=0)

    # Separating line between top and bottom
    ax.axvline(x=top_n - 0.5, color='gray', linestyle='--')

    # Annotate top/bottom
    ymax = max(combined[metric]) * 1.05
    ax.text(top_n // 2, ymax, "TOP TARGETS", ha='center', fontweight='bold', color='tab:blue')
    ax.text(top_n + top_n // 2, ymax, "BOTTOM TARGETS", ha='center', fontweight='bold', color='tab:orange')

    # Style
    ax.set_title(f'Target Performance for {config_str}\nMetric: {metric.replace("_mean", "").upper()}', fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace('_mean', '').upper())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_ylim(0, max(combined[metric]) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title="Target Group")

    plt.tight_layout()
    file_path = output_path + "by_targets/"
    ensure_directory_exists(file_path)
    plt.savefig(file_path+f'{model_name}_ctx={kw_context}_vl={vl_propform}top_btm_{top_n}_targets.pdf', bbox_inches='tight') if output_path else plt.show()

def groupbarplot_stacked(df, model_col="model", target_col="target", metric_col="acc_mean", figsize=(16, 12), output_path=OUTPUT_PATH):
    ensure_directory_exists(OUTPUT_PATH)
    sns.set(style="whitegrid")
    sns.set_context("talk", font_scale=1.6)

    fig, axes = plt.subplots(nrows=2, figsize=figsize, sharey=True)

    for ax, target_list, key in zip(axes, [SUBSAMPLE_PARA, SUBSAMPLE_SYNTAGM], ["Paradigmatic", "Syntagmatic"]):
        df_sub = df[df[target_col].isin(target_list)].copy()

        # 可选替换 target 名称（缩短）
        if key == "Paradigmatic":
            df_sub[target_col] = df_sub[target_col].replace({
                "Dérivation_adverbiale": "Deriv_Adv",
                "Dérivation_adjectivale": "Deriv_Adj",
                "Subst_sens_similaire": "Subst_Sim"
            })
        else:
            df_sub[target_col] = df_sub[target_col].replace({
                "Colloc_préposition": "Colloc_Prep",
                "Collocation_verbale": "Colloc_Verb"
            })

        target_order = df_sub.groupby(target_col)[metric_col].mean().sort_values(ascending=False).index.tolist()

        sns.barplot(
            data=df_sub,
            x=target_col,
            y=metric_col,
            hue=model_col,
            order=target_order,
            palette=MODEL_PALETTE,
            errorbar=None,
            ax=ax
        )
        ax.set_xlabel(f"{key} LFs")
        ax.set_ylabel(metric_col.replace("_", " ").title())
        ax.tick_params(axis='x', rotation=90)
        ax.get_legend().remove()

    fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.53),  # y≈0.53 控制上下居中，x=0.5表示水平居中
        ncol=3,  # 横排
        frameon=False,
    )
    plt.tight_layout(h_pad=3)  # 增加子图之间的距离
    ensure_directory_exists(output_path + "by_targets/")
    plt.savefig(output_path+'by_targets/sampled_targets.pdf', format='pdf', bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse model performance by configuration and targets.")
    parser.add_argument('--input_csv', type=str, default=RES)
    parser.add_argument('--input_best', type=str, default='experiments/task_binary_3/output/best_results.csv')
    parser.add_argument('--output_path', type=str, default=OUTPUT_PATH)
    args = parser.parse_args()
    # Load and prepare data
    df = load_multiple_csv(args.input_csv)
    df = prepare_data(df)

    # Calculate metrics by model and k-shot
    matric_all = calculate_metrics_by_model_kshot_target(df)
    # Save the calculated metrics
    matric_all.to_csv(args.output_path + 'metrics_by_model_target&config.csv', index=False)
    # Plot model performance
    
    plot_model_perf_by_config_targets(matric_all, model_name='Mistral', k_shot=10, kw_context=0, vl_propform=0, 
                                         metric='acc_mean', var_metric='var_by_acc', top_n=15,
                                         output_path=args.output_path)
    plot_model_perf_by_config_targets(matric_all, model_name='Qwen', k_shot=10, kw_context=0, vl_propform=1,
                                         metric='acc_mean', var_metric='var_by_acc', top_n=15,
                                         output_path=args.output_path)
    plot_model_perf_by_config_targets(matric_all, model_name='LLaMA', k_shot=10, kw_context=0, vl_propform=1,
                                         metric='acc_mean', var_metric='var_by_acc', top_n=15,
                                         output_path=args.output_path)
    # 像之前的 overall_perf.py 一样，提取最佳配置
    matric_overall = calculate_metrics_by_model_kshot(df)
    best_config = extract_best_config(matric_overall)
    print("Best Configurations:", best_config)
    # Filter best results based on the best configuration
    best_perf = filter_best_results(matric_all, best_config)
    best_perf.to_csv(args.output_path + 'best_perfs.csv', index=False)
    # Grouped bar plot for sampled LFs
    groupbarplot_stacked(best_perf, model_col='model', target_col='target', metric_col='acc_mean', figsize=(16, 12), output_path=args.output_path)
