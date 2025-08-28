# ======================= IMPORTS =======================
import os
import sys
import re
import csv
import time
import logging
import argparse
import pandas as pd
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)

from generate_prompt import load_csv, load_json, generate_prompt, QUESTION_PATH, DESCRPTIONS_PATH, PROMPT_FIX

# ======================= CONSTANTS =======================
QWEN = "Qwen/Qwen2.5-14B-Instruct-1M"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
LLAMA3 = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODELS = {
    "qwen": QWEN,
    "mistral": MISTRAL,
    "llama3": LLAMA3
}

# TASK_DEFINITION = """Tu es un expert des Fonctions Lexicales. Une fonction lexicale est un concept linguistique qui représente un certain type de relation entre mots, en produisant pour chaque mot-clef un autre mot (sa valeur) qui réalise certaines contraintes, comme changer son sens ou sa catégorie grammaticale. A partir d'une description d'un type de fonction lexicale ciblé et de quelques exemples résolus, tu dois déterminer si les autres exemples fournis par l'utilisateur correspondent au type de fonction lexicale ciblé. Pour chaque paire, l'utilisateur va fournir: 1) la paire mot-clef -> valeur, 2) la forme propositionnelle du mot-clef qui indique la numérotation de ses actants, 3) un exemple d'utilisation du mot-clef. """

SIMPLE_RES = "Tu ne dois répondre que par '**Oui**' ou '**Non**', sans explication supplémentaire. \n"

AVEC_JSTF = (
    "Tu dois répondre d'abord par '**Oui**' ou '**Non**', puis expliquer pourquoi la paire de mots correspond ou non à la fonction lexicale.\n"
    "Ta réponse doit toujours contenir '**Oui**' ou '**Non**' avec ce format précis, entre crochets spéciaux. Puis, tu peux expliquer ta réponse en utilisant des phrases complètes.\n"
)

# ======================= FUNCTIONS =======================
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(model_name: str, use_4bit: bool = True):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        local_files_only=True
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=config,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def setup_files(model_name, batch_id):
    # 提取前缀时间戳作为统一目录名
    batch_dir = batch_id.split('_')[0]  # 如 "20240515_231501"

    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(PARENT_DIR, "logs", batch_dir)
    results_dir = os.path.join(PARENT_DIR, "res", batch_dir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_short_name = next(
        (k for k, v in MODELS.items() if v == model_name),
        model_name.split('/')[-1].split('-')[0]
    )

    log_file = os.path.join(log_dir, f"{model_short_name}_{batch_id}.log")
    results_file = os.path.join(results_dir, f"{model_short_name}_{batch_id}.csv")

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    logging.info(f"Starting evaluation run with model: {model_name}, batch: {batch_id}")

    return results_file, log_file, batch_id

def save_results(results, filename):
    if not results:
        return
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "target", "question_id", "scope_id", "kw_context", "vl_propform", "justify", "k-shot", "repeat_num", "q_kw", "q_vl", "q_lf", "expected", "response"])
        writer.writerows(results)

def extract_response_std(resp: str) -> str:
    lines = [line.strip() for line in resp.strip().splitlines() if line.strip()]
    first_two = lines[:2]

    for line in first_two:
        for pattern in [r"\*\*(Oui|Non)\*\*", r"[【\[]\s*(Oui|Non)\s*[】\]]", r"^(Oui|Non)\b", r"^(?:[A-Za-z\s]*[:：])?\s*\*\*(Oui|Non)\*\*",
            r"^(?:[A-Za-z\s]*[:：])?\s*[【\[]\s*(Oui|Non)\s*[】\]]",
            r"^(?:[A-Za-z\s]*[:：])?\s*(Oui|Non)\b"]:
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                return match.group(1).capitalize()

    return "Unknown"

# ======================= MAIN FUNCTION =======================
def ask_model_batch(
    model,
    tokenizer,
    questions: dict,
    descriptions: pd.DataFrame,
    k: int = 2,
    target_list: list = None,
    scope_size: int = None,
    seed: int = 42,
    kw_context: bool = False,
    vl_propform: bool = False,
    justification: bool = False,
    repeat_num: int = 5,
    batch_id: str = None,
    batch_size: int = 10,
    model_name: str = "",
    max_new_tokens: int = None,
):
    # Max new tokens
    if max_new_tokens is None:
        max_new_tokens = 256 if justification else 16
    model_short_name = model_name.split('/')[0]
    results_file, log_file, batch_id = setup_files(model_name, batch_id)
    set_seed(seed)
    results = []
    start_time_all = time.time()

    targets = target_list if target_list else list(questions.keys())

    for target in targets:
        system_prompt = PROMPT_FIX + (AVEC_JSTF if justification else SIMPLE_RES)
        log_block = f"""\n================ SYSTEM PROMPT ===================\nModel: {model_short_name}\nKw_ctx: {int(kw_context)} | Vl_prop: {int(vl_propform)} | Justification: {int(justification)}\nSystem Prompt:\n{system_prompt.strip()}\n=================================================="""
        logging.info(log_block)
        questions_space = [
            q for q in questions[target]
            if scope_size is None or q.get("scope", 999) <= scope_size
        ]
        for q_id, question_data in enumerate(questions_space, start=1):
            prompt, question = generate_prompt(question_data=question_data,
                    descriptions=descriptions,
                    scope_size=scope_size,
                    k=k,
                    target=target,
                    kw_context=kw_context,
                    vl_propform=vl_propform,
                    justification=justification)
            question_data = questions[target][q_id - 1]
            repeat_responses = []
            prompt_batch, metadata_batch = [], []

            #prepare something for logging
            pos_ex_paires_kw = [p["keyword"] for p in question_data["ex_pos_prompt"]]
            pos_ex_paires_val = [p["value"] for p in question_data["ex_pos_prompt"]]
            pos_ex_paires = list(zip(pos_ex_paires_kw, pos_ex_paires_val))[:int(k/2)]
            neg_ex_paires_kw = [p["keyword"] for p in question_data["ex_neg_prompt"]]
            neg_ex_paires_val = [p["value"] for p in question_data["ex_neg_prompt"]]
            neg_ex_paires = list(zip(neg_ex_paires_kw, neg_ex_paires_val))[:int(k/2)]

            for r in range(repeat_num):
                set_seed(seed + r)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + "QUESTION:\n\n" + question},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_batch.append(text)
                meta_info = question_data["ex_question"]
                metadata_batch.append((target, q_id, k, r, meta_info, question_data.get("scope", ""),question_data.get("expected", "")))

                if len(prompt_batch) >= batch_size or (r == repeat_num - 1):
                    inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
                    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
                    generated_ids = [
                        gen[len(inp):] for inp, gen in zip(inputs["input_ids"], generated)
                            ]
                    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    for i, resp in enumerate(responses):
                        response_std = extract_response_std(resp)
                        if response_std == "Unknown":
                            logging.warning(f"Unknown response for {model_short_name} on {target} Q{q_id} #{r+1}: \n{resp}")
                            continue
                        meta = metadata_batch[i]
                        row = [model_short_name, # model: qwen, mistral, llama3
                               meta[0], # target
                               meta[1], # question_id
                               meta[5], # scope_id
                               int(kw_context), # kw_context gived ?
                               int(vl_propform), # vl_propform gived ?
                               int(justification), # justification
                               meta[2], # k-shot
                               meta[3]+1, # repeat_num
                               meta[4].get("keyword", ""), # question keyword
                               meta[4].get("value", ""),  # question value
                               meta[4].get("lf_name", ""), # question lf_name
                               meta[6], # expected
                               response_std]
                        results.append(row)
                        repeat_responses.append((meta[3]+1, resp))
                        print(f"[{meta[0]} Q{meta[1]} #{meta[3]+1}] {meta[4].get('keyword', '')}->{meta[4].get('value', '')} | Label: {meta[6]} | Response: {resp}")

                    save_results(results, results_file)
                    prompt_batch, metadata_batch, results = [], [], []
            # 每个问题日志一次 + 所有 repeat response
            log_lines = [
                f"\n---------- Target: {target} ----------",
                f"> Q{q_id} | Scope: {question_data.get('scope', '')} | True LF: {question_data['ex_question'].get('lf_name', '')}",
                "USER PROMPT:",
                prompt.strip(),
                "",
                "QUESTION:",
                question.strip(),
                f"Expected: {question_data.get('expected', '')}",
                "~~~~~~~~~~~~~~~~~~~~ RESPONSE ~~~~~~~~~~~~~~~~~~~~",
                "# | Response (raw) by repeat",
                "--|----------------"
            ]
            log_lines += [f"{resp_num} | {resp.strip()}" for resp_num, resp in repeat_responses]
            log_lines += [
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
                "--------------------------------------------------"
            ]
            logging.info('\n'.join(log_lines))
    elapsed = time.time() - start_time_all
    print(f"\nAll done in {elapsed:.2f}s")

# ======================= CLI =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, choices=MODELS.keys(), default='qwen')
    parser.add_argument('--questions_path', type=str,  default=QUESTION_PATH)
    parser.add_argument('--descriptions_path', type=str, default=DESCRPTIONS_PATH)
    parser.add_argument('-tg', '--targets', type=str, nargs='+')
    parser.add_argument('-k', '--k', type=int, default=2)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-r', '--repeat_num', type=int, default=5)
    parser.add_argument('--scope_size', type=int, default=None)
    parser.add_argument('--batch_id', type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--kw_context', action='store_true')
    parser.add_argument('--vl_propform', action='store_true')
    parser.add_argument('--justification', action='store_true')
    args = parser.parse_args()

    questions = load_json(args.questions_path)
    descriptions = load_csv(args.descriptions_path)
    model_path = MODELS[args.model]
    model, tokenizer = load_model(model_path)

    try:
        ask_model_batch(
        model=model,
        tokenizer=tokenizer,
        model_name=MODELS[args.model],
        questions=questions,
        descriptions=descriptions,
        target_list=args.targets,
        k=args.k,
        seed=args.seed,
        repeat_num=args.repeat_num,
        scope_size=args.scope_size,
        batch_id=args.batch_id,
        batch_size=args.batch_size,
        kw_context=args.kw_context,
        vl_propform=args.vl_propform,
        justification=args.justification
        )
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()
