# ======================= IMPORTS =======================
# Basic imports
import os
import sys
import json
import argparse
import pandas as pd

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)  # Script runs from project root directory

# questions prepared
QUESTION_PATH = "experiments/task_binary_3/data/questions.json"
DESCRPTIONS_PATH = "data/interim/lfs-descriptions-v3.csv"
PROMPT_FIX = """Tu es un linguiste expert des fonctions lexicales de la théorie Sens-Texte. Je vais te fournir la définition d'une famille de fonctions lexicales, suivie d'exemples positifs et négatifs. Ensuite, je te soumettrai une paire de lexies. Tu devras répondre par '**Oui**' si cette paire correspond au lien lexical modélisé par cette famille de fonctions lexicales, ou répondre par '**Non**' dans le cas contraire. Pour chaque mot-clé, je te fournirai un contexte au format KWIC (Key Word In Context) qui te permettra de cerner l'acception qui nous intéresse, ainsi que la forme propositionnelle de la lexie (qui indique sa nature prédicative). Dans la forme propositionnelle, les premier, deuxième, troisième, etc. arguments sont représentés par $1, $2, $3, etc. \n"""

# ======================= HELPER FUNCTIONS =======================
def load_csv(file_path):
    """Load CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path, encoding="utf-8")
    return df
def load_json(file_path):
    """Load questions from a json file."""
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    return questions

def generate_prompt(
    question_data: dict,
    descriptions: pd.DataFrame,
    target: str,
    k: int = 10,
    kw_context: bool = False,
    vl_propform: bool = False,
    justification: bool = False,
    scope_size: int = None,  # optional, just for logging
):
    # 目标描述
    target_description = descriptions[descriptions["node"] == target.rstrip("*")]["description"].values[0]
    if target_description == "":
        print(f"Warning: No description found for target {target}.")

    prompt = f"{target_description}\n\n"
    prompt += f"Voici des exemples positifs de cette classe:\n\n"
    for ex in question_data["ex_pos_prompt"][:k // 2]:
        prompt += f"{ex['keyword']} -> {ex['value']}\n"
        prompt += f"Forme propositionnelle du mot-clé: {ex['kw_propform']}\n"
        if kw_context:
            prompt += f"Contexte KWIC du mot-clé: {ex['kw_context']}\n"
        if vl_propform:
            prompt += f"Forme propositionnelle de la valeur: {ex['vl_propform']}\n"
        prompt += "Réponse: **Oui**\n"
        if justification:
            prompt += "Justification : [parce que ...]\n"
        prompt += "\n"

    prompt += f"Voici des exemples négatifs de cette classe:\n\n"
    for ex in question_data["ex_neg_prompt"][:k // 2]:
        prompt += f"{ex['keyword']} -> {ex['value']}\n"
        if kw_context:
            prompt += f"Exemple d'utilisation du mot-clé: {ex['kw_context']}\n"
        prompt += f"Forme propositionnelle du mot-clé: {ex['kw_propform']}\n"
        if vl_propform:
            prompt += f"Forme propositionnelle de la valeur: {ex['vl_propform']}\n"
        prompt += "Réponse: **Non**\n"
        if justification:
            prompt += "Justification : [parce que ...]\n"
        prompt += "\n"

    ex_q = question_data["ex_question"]
    quest_text = f"{ex_q['keyword']} -> {ex_q['value']}\n"
    if kw_context:
        quest_text += f"Exemple d'utilisation du mot-clé: {ex_q['kw_context']}\n"
    quest_text += f"Forme propositionnelle du mot-clé: {ex_q['kw_propform']}\n"
    if vl_propform:
        quest_text += f"Forme propositionnelle de la valeur: {ex_q['vl_propform']}\n\n"

    if justification:
        quest_text += "Est-ce que la paire de mots ci-dessus constituent aussi un exemple de cette classe de fonction lexicale? Réponds d'abord par '**Oui**' ou par '**Non**', puis justifie ta réponse en expliquant pourquoi.\n"
    else:
        quest_text += "Est-ce que la paire de mots ci-dessus constituent aussi un exemple de cette classe de fonction lexicale? Réponds uniquement par '**Oui**' ou par '**Non**'.\n"

    return prompt, quest_text

       




def main():
    # ======================= ARGPARSE =======================
    parser = argparse.ArgumentParser(description="Generate prompts for the model.")
    parser.add_argument("--questions_path", type=str, default=QUESTION_PATH, help="Path to the questions file.")
    parser.add_argument("--descriptions_path", type=str, default=DESCRPTIONS_PATH, help="Path to the descriptions file.")
    parser.add_argument("--target", type=str, required=True, help="Target node.")
    parser.add_argument("--question_id", type=int, required=True, help="Question ID.")
    parser.add_argument("--scope_size", type=int, default=None, help="Scope number.")
    parser.add_argument("--k", type=int, default=10, help="Number of examples.")
    parser.add_argument("--kw_context", action='store_true', help="Include keyword context.")
    parser.add_argument("--vl_propform", action='store_true', help="Include value proposition form.")
    parser.add_argument("--justification", action='store_true', help="Include justification.")
    args = parser.parse_args()

    # ======================= MAIN LOGIC =======================
    questions = load_json(args.questions_path)
    descriptions = load_csv(args.descriptions_path)
    # print(f"args.target = {args.target}, type = {type(args.target)}")
    prompt, quest_text = generate_prompt(
    questions=questions,
    descriptions=descriptions,
    target=args.target,
    question_id=args.question_id,
    scope_size=args.scope_size,
    k=args.k,
    kw_context=args.kw_context,
    vl_propform=args.vl_propform,
    justification=args.justification
    )
    
    print(prompt)
    print(quest_text)

if __name__ == "__main__":
    main()
