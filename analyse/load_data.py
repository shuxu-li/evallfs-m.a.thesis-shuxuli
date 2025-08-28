
import glob
import pandas as pd

# ============== Constants ===============
RES = 'experiments/task_binary_3/res/20250629/*.csv'
OUTPUT_PATH = 'experiments/task_binary_3/output/'
MODEL_PALETTE = {
    "Qwen": "#1f77b4",     # blue
    "LLaMA": "#ff7f0e",    # orange
    "Mistral": "#2ca02c"   # green
}

SUBSAMPLE_PARA = ["Dérivation_adverbiale", "Dérivation_adjectivale", "V0",  "S0", "A0", "Sloc", "Adv0", "S_i", "S2", "Qual1"]

SUBSAMPLE_SYNTAGM = ["Colloc_préposition", "Collocation_verbale", "Modificateur", "Ver", "Loc", "Oper2", "Fact_i", "Real2", "Func2"]
# =========== Helper Functions ===========

def load_multiple_csv(path_pattern):
    files = glob.glob(path_pattern)
    dfs = [pd.read_csv(file) for file in files if file.endswith('.csv')]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def prepare_data(df):
    """
    Prepare the data for analysis.
    """
    try:
        # add Q_type column
        df['Q_type'] = df.apply(lambda x: 'Q_pos' if x['expected'] == 'Oui' else 'Q_neg', axis=1)

        # rename models
        md_mapping = {
            "meta-llama": "LLaMA",
            "mistralai": "Mistral",
            "Qwen": "Qwen",
        }
        df['model'] = df['model'].replace(md_mapping)

        # prepare true and predicted labels
        df["True"] = df["expected"].str.lower() == "oui"
        df["Pred"] = df["response"].str.lower() == "oui"

        return df
    except Exception as e:
        raise RuntimeError(f"Error during data preparation: {e}")


# def select_best_results(df, model_col='model'):
#     """
#     Select the best results for each model based on the highest accuracy.
#     """
#     best_results = []
#     for model, config in MODEL_BEST_CONFIG.items():
#         model_df = df[df[model_col] == model]
#         if model_df.empty:
#             continue
#         try:
#             # Filter by context, vfp, and k
#             filtered_df = model_df[
#                 (model_df['ctx'] == config['ctx']) &
#                 (model_df['vfp'] == config['vfp']) &
#                 (model_df['k'] == config['k'])
#             ]
#             if filtered_df.empty:
#                 continue

#             best_row = filtered_df.loc[filtered_df['accuracy'].idxmax()]
#             best_results.append(best_row)
#         except Exception as e:
#             continue

#     return pd.DataFrame(best_results) if best_results else pd.DataFrame()
