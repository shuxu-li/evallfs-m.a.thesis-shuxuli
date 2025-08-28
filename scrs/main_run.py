# Combine all parameters and run jobs on 2 GPUs in parallel
import itertools
import subprocess
import time
from datetime import datetime
from multiprocessing import Process

# Constants
QUESTIONS_PATH = "experiments/task_binary_3/data/questions-2.json"
DESCRIPTIONS_PATH = "data/interim/lfs-descriptions-v3.csv"
REPEAT = 5
SEED = 42
MAX_NEW_TOKENS = 32

# Batch size for each model
# Qwen will occupy more memory, so we assign it a smaller batch size
# Mistral and Llama3 can use larger batch sizes
MODEL_BATCH_SIZE = {
    "qwen": 4,
    "mistral": 8,
    "llama3": 8
}
# targets = ["Substitutive", "Subst_sens_similaire", "Gener", "Dérivation_nominale", "S0", "N_dériv_ajout_de_sens",  "S_i", "S2", "Dérivation_adjectivale", "A0", "A1", "Able_i", "Able1", "Qual_i", "Modificateur", "Modif_ajout_de_sens", "Magn",  "Verbe_support", "Func_i", "Oper_i", "Func0", "Oper1", "V_collocation_ajout_de_sens", "Vreal", "Real_i", "Fact_i", "Fact0", "Real2"]

# Combinations of all parameters: models, k, kw_context, vl_propform
models = ["qwen", "mistral", "llama3"]
ks = [2, 6, 10]
kw_context_opts = [True, False]
vl_propform_opts = [True, False] # We only use vl_propform for the positive examples
jus_opts = [False, True] # Justification is only used for the positive examples

# Generate all combinations of parameters
all_jobs = list(itertools.product(models, ks, kw_context_opts, vl_propform_opts, jus_opts))

# Generate batch IDs for each model, because we want to save the results in a unique file for each model
batch_ids = {m: datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{m}" for m in models}

#==========Special batchIDs==========
# batch_ids = {m: f"{m}_test_shortsys" for m in models}


# Separate jobs for Qwen and other models
# Beacause Qwen will take more memory and more time
qwen_jobs = [job for job in all_jobs if job[0] == "qwen"]
other_jobs = [job for job in all_jobs if job[0] in {"mistral", "llama3"}]


def run_jobs_on_gpu(job_list, gpu_id):
    for idx, (model, k, kw_ctx, vl_prop, justification) in enumerate(job_list, 1):  
        batch_id = batch_ids[model]
        batch_size = MODEL_BATCH_SIZE[model]

        PYTHON_EXEC = "/home/shuxu/lmeval/bin/python3"
        MAIN_SCRIPT = "experiments/task_binary_3/scrs/ask.py"
        cmd = [
            PYTHON_EXEC, MAIN_SCRIPT,  
            "--model", model,
            "--questions_path", QUESTIONS_PATH,
            "--descriptions_path", DESCRIPTIONS_PATH,
            "--k", str(k),
            "--repeat_num", str(REPEAT),
            "--batch_size", str(batch_size),
            "--seed", str(SEED),
            "--batch_id", batch_id,
            # "--targets", *targets  # Updated to include targets
        ]
        if kw_ctx:
            cmd.append("--kw_context")
        if vl_prop:
            cmd.append("--vl_propform")
        if justification:
            cmd.append("--justification")

        print(f"[GPU{gpu_id}] ▶️ {model} | k={k} | ctx={kw_ctx} | prop={vl_prop} | batch={batch_size}")
        start = time.time()
        subprocess.run(cmd, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
        elapsed = time.time() - start
        print(f"[GPU{gpu_id}] ✅ Done in {elapsed:.2f}s\n")
        time.sleep(1)


if __name__ == "__main__":
    print(f"Running {len(all_jobs)} jobs on 2 GPUs in parallel\n")
    p0 = Process(target=run_jobs_on_gpu, args=(qwen_jobs, 0))
    p1 = Process(target=run_jobs_on_gpu, args=(other_jobs, 1))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    print("✅ All jobs finished!")
