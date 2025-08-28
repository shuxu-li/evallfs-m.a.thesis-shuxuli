# overall_performance.py
import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from load_data import load_multiple_csv, prepare_data, RES, OUTPUT_PATH, MODEL_PALETTE


def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# calculate confusion matrix
def evaluate_binary_predictions(df, groupby_cols=None):
    if groupby_cols:
        grouped = df.groupby(groupby_cols)
    else:
        grouped = [("Global", df)]

    for group_key, group_df in grouped:
        print("="*80)
        print(f"🔹 Results for: {group_key}")
        y_true = group_df["expected"]
        y_pred = group_df["response"]
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Oui", "Non"], digits=3))

def plot_confusion_matrices_by_model(
    df: pd.DataFrame,
    expected_col: str = "expected",
    response_col: str = "response",
    model_col: str = "model",
    labels: list = ["Oui", "Non"],
    label_names: list = ["Yes", "No"],
    savepath: str = "otpt/confusion_matrices_by_model.pdf"
):
    """
    对每个模型分别绘制其 confusion matrix 热力图，并保存为 PDF。
    
    Parameters:
    - df: 包含预测结果的 DataFrame。
    - expected_col: 真值列名。
    - response_col: 预测值列名。
    - model_col: 模型名称列名。
    - labels: confusion_matrix 的标签顺序。
    - label_names: 用于图中显示的标签名称（如英文化）。
    - savepath: PDF 保存路径。
    """
    models = df[model_col].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3))

    if n_models == 1:
        axes = [axes]  # 保证是可迭代的

    sns.set_context("paper", font_scale=1.3)

    for ax, model_name in zip(axes, models):
        sub_df = df[df[model_col] == model_name]

        cm = confusion_matrix(sub_df[expected_col], sub_df[response_col], labels=labels)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax
        )

        ax.set_title(f"{model_name}")
        ax.set_xlabel("Prediction", fontsize=12)
        ax.set_ylabel("Truth", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.show()

def calculate_metrics_by_model_kshot(df, 
                                     true_col='expected', pred_col='response',
                                     model_col='model', kshot_col='k-shot',
                                     repeat_col='repeat_num',
                                     decimal_places=3,
                                     context_col='kw_context', 
                                     propform_col='vl_propform',
                                     output_path=OUTPUT_PATH):
    ensure_directory_exists(output_path)
    results_list = []

    group_cols = [model_col, kshot_col, context_col, propform_col]
    for (model, kshot, kw_context, vl_propform), group in df.groupby(group_cols):
        metrics = {
            'model': model,
            'k-shot': kshot,
            'kw_context': kw_context,
            'vl_propform': vl_propform
        }

        acc_values, prec_values, rec_values, f1_values = [], [], [], []

        for repeat in sorted(group[repeat_col].unique()):
            repeat_group = group[group[repeat_col] == repeat]
            acc = accuracy_score(repeat_group[true_col], repeat_group[pred_col])
            prec = precision_score(repeat_group[true_col], repeat_group[pred_col], average='weighted', zero_division=0) 
            rec = recall_score(repeat_group[true_col], repeat_group[pred_col], average='weighted', zero_division=0)
            f1 = f1_score(repeat_group[true_col], repeat_group[pred_col], average='weighted', zero_division=0)

            acc_values.append(acc)
            prec_values.append(prec)
            rec_values.append(rec)
            f1_values.append(f1)

        metrics.update({
            'acc_mean': round(np.mean(acc_values), decimal_places),
            'prec_mean': round(np.mean(prec_values), decimal_places),
            'rec_mean': round(np.mean(rec_values), decimal_places),
            'f1_mean': round(np.mean(f1_values), decimal_places),
            'var_by_acc': round(np.var(acc_values, ddof=1), 6),
            'var_by_f1': round(np.var(f1_values, ddof=1), 6),
        })

        results_list.append(metrics)

    results_df = pd.DataFrame(results_list)
    columns_order = [
        'model', 'k-shot', 'kw_context', 'vl_propform',
        'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean',
        'var_by_acc', 'var_by_f1'
    ]
    results_df = results_df[columns_order]
    results_df.to_csv(output_path+'model_mtrc_by_config.csv', index=False)
    print(f"Metrics by model and k-shot saved to {output_path}")
    return results_df


def extract_best_config(metrics_df, model_col='model', kshot_col='k-shot', context_col='kw_context', propform_col='vl_propform'):
    """
    Extract the best configuration (k, ctx, propform) for each model based on the highest accuracy.
    """
    best_config = {}
    for model in metrics_df[model_col].unique():
        model_df = metrics_df[metrics_df[model_col] == model]
        if model_df.empty:
            continue
        best_row = model_df.loc[model_df['acc_mean'].idxmax()]
        best_config[model] = {
            'k': int(best_row[kshot_col]),
            'ctx': int(best_row[context_col]),
            'propform': int(best_row[propform_col])
        }
    return best_config


def filter_best_results(df, best_config, model_col='model', kshot_col='k-shot', context_col='kw_context', propform_col='vl_propform'):
    """
    Filter the results based on the best configuration for each model.
    """
    filtered_results = []
    for model, config in best_config.items():
        model_df = df[
            (df[model_col] == model) &
            (df[kshot_col] == config['k']) &
            (df[context_col] == config['ctx']) &
            (df[propform_col] == config['propform'])
        ]
        filtered_results.append(model_df)
    return pd.concat(filtered_results, ignore_index=True) if filtered_results else pd.DataFrame()



# 画出trend图
def plot_metrics_trends(df, metric='f1_mean', figsize=(8, 6), output_path=OUTPUT_PATH):
    assert metric in ['f1_mean', 'acc_mean'], "Metric must be 'f1_mean' or 'acc_mean'"

    models = df['model'].unique()
    kw_contexts = df['kw_context'].unique() # 0/1
    vl_props = df['vl_propform'].unique() # 0/1

    # paterns
    # palette = dict(zip(models, sns.color_palette("tab10", len(models))))
    palette = MODEL_PALETTE
    dashes = dict(zip(kw_contexts, ['solid', 'dashed', 'dotted', 'dashdot']))
    markers = dict(zip(vl_props, ['o', 's', '^', 'D', 'P', '*']))

    plt.figure(figsize=figsize)

    # plot by combining model, kw_context, and vl_propform
    for model in models:
        for kw_ctx in kw_contexts:
            for vl_prop in vl_props:
                subset = df[(df['model'] == model) &
                            (df['kw_context'] == kw_ctx) &
                            (df['vl_propform'] == vl_prop)]
                if subset.empty:
                    continue

                label = f"{model} | ctx={kw_ctx} | prop={vl_prop}"
                sns.set_style("whitegrid")
                sns.set_context("paper", font_scale=1.3)
                sns.lineplot(
                    data=subset,
                    x='k-shot',
                    y=metric,
                    label=label,
                    color=palette[model],
                    linestyle=dashes[kw_ctx],
                    marker=markers[vl_prop],
                    markersize=8,
                    linewidth=2,                )

    plt.title(f'Model Performance ({metric.replace("_mean", "").upper()})')
    plt.xlabel('K-shot')
    plt.ylabel(metric.replace('_mean', '').upper())
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Model | kw_context | vl_propform', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_path}/overall/{metric}_trends.pdf', format='pdf', bbox_inches='tight')
    # plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse overall performance of binary classification models.")
    parser.add_argument('--data_path', type=str, default=RES, help="Path to the results CSV files.")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_PATH, help="Directory to save output files.")
    args = parser.parse_args()

    # 1. 加载和预处理数据
    df = load_multiple_csv(args.data_path)
    df = prepare_data(df)
    # ensure_directory_exists(args.output_dir)
    # 计算confusion matrix
    evaluate_binary_predictions(df)
    plot_confusion_matrices_by_model(df, 
                                     expected_col='expected',
                                     response_col='response',
                                     model_col='model',
                                     labels=['Oui', 'Non'],
                                     label_names=['Yes', 'No'],
                                     savepath=OUTPUT_PATH + 'cnfsn_mtrx_by_models.pdf')

    # 2. 计算各配置下的性能指标
    metrics_df = calculate_metrics_by_model_kshot(df, output_path=args.output_dir)

    # 3. 提取每个模型 accuracy 最优的配置（k-shot, kw_context, vl_propform）
    best_config = extract_best_config(metrics_df)
    print("Best Configurations:", best_config)

    # 4. 按照最佳配置筛选原始结果
    best_results = filter_best_results(df, best_config)

    # 5. 保存筛选后的结果
    best_results_path = os.path.join(args.output_dir, 'best_results.csv')
    best_results.to_csv(best_results_path, index=False)
    print(f"Best results saved to {best_results_path}")
    # 6. 绘制趋势图
    plot_metrics_trends(metrics_df, metric='f1_mean')
    plot_metrics_trends(metrics_df, metric='acc_mean')