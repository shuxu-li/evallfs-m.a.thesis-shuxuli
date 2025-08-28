# run_qwen_only.py
from main_run import run_jobs_on_gpu, MODEL_BATCH_SIZE

# Set the parameters for Qwen
qwen_jobs = [("qwen", k, ctx, prop, justification)
              for k in [2, 6, 10]
              for ctx in [True, False]
              for prop in [True, False]
              for justification in [False]]

print(f"🚀 Running {len(qwen_jobs)} Qwen jobs on GPU 0")
run_jobs_on_gpu(qwen_jobs, gpu_id=0)