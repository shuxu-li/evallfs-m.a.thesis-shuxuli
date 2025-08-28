# run_others_only.py
from main_run import run_jobs_on_gpu, MODEL_BATCH_SIZE

# 构建 mistral 和 llama3 的任务组合
other_models = ["mistral", "llama3"]
other_jobs = [(model, k, ctx, prop, justification)
              for model in other_models
              for k in [2, 6, 10]
              for ctx in [True]
              for prop in [True, False]
              for justification in [False]]

print(f"🚀 Running {len(other_jobs)} jobs (Mistral & LLaMA3) on GPU 1")
run_jobs_on_gpu(other_jobs, gpu_id=1)
