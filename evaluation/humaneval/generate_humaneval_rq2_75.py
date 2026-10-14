import json
import os
import time
from peft import PeftModel

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from evalplus.data import get_human_eval_plus


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "/mnt/d/CFLLM/models/Qwen2.5-Coder-7B-Base"

ADAPTER_PATH = (
    "/mnt/d/CFLLM/outputs/rq2/"
    "qwen_base_pycpp75/checkpoint-epoch-3"
)

OUTPUT_DIR = "/mnt/d/CFLLM/outputs/humaneval"

RAW_PATH = os.path.join(
    OUTPUT_DIR,
    "qwen2.5-coder-7b-rq2-pycpp75-epoch3_raw.jsonl"
)

EVAL_PATH = os.path.join(
    OUTPUT_DIR,
    "qwen2.5-coder-7b-rq2-pycpp75-epoch3_eval.jsonl"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Reproduction settings
# ============================================================

BASE_SEED = 42

# Paper-specified evaluation temperature
TEMPERATURE = 0.6

# Reimplementation choices not explicitly specified in paper
MAX_NEW_TOKENS = 512
TOP_P = 1.0
TOP_K = 0


# ============================================================
# Conservative structural post-processing
# ============================================================

def truncate_completion(completion: str) -> str:
    """
    Keep only code belonging to the target function body.

    HumanEval prompt already contains the top-level function signature
    and docstring. Therefore its continuation should be indented.

    Once a non-empty generated line returns to column 0, generation has
    left the requested function. Everything after that point is removed.

    No semantic repair is performed.
    """

    kept = []

    for line in completion.splitlines(keepends=True):

        # Preserve blank lines.
        if not line.strip():
            kept.append(line)
            continue

        indent = len(line) - len(line.lstrip(" \t"))

        # A non-empty top-level line is outside the target function.
        if indent == 0:
            break

        kept.append(line)

    cleaned = "".join(kept).rstrip()

    if cleaned:
        cleaned += "\n"

    return cleaned


# ============================================================
# Resume support
# ============================================================

def load_completed_tasks(path):
    completed = set()

    if not os.path.exists(path):
        return completed

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                completed.add(obj["task_id"])
            except Exception:
                pass

    return completed


def rebuild_eval_file(raw_path, eval_path):
    """
    Rebuild EvalPlus input from the audit/raw file.

    This prevents raw and evaluation files from becoming inconsistent
    after an interrupted run.
    """

    with open(raw_path, "r", encoding="utf-8") as fin, \
         open(eval_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            record = {
                "task_id": obj["task_id"],
                "solution": obj["solution"],
            }

            fout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


# ============================================================
# Environment
# ============================================================

print("===== DEVICE =====")
print("CUDA available:", torch.cuda.is_available())
print("visible GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Dataset
# ============================================================

print("\n===== HUMANEVAL+ =====")

problems = get_human_eval_plus()

task_ids = sorted(
    problems.keys(),
    key=lambda x: int(x.split("/")[-1])
)

print("number of tasks:", len(task_ids))

if len(task_ids) != 164:
    raise RuntimeError(
        f"Expected 164 HumanEval tasks, got {len(task_ids)}"
    )


# ============================================================
# Model
# ============================================================

print("\n===== TOKENIZER =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


print("\n===== MODEL =====")

load_start = time.time()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print("loading LoRA adapter:", ADAPTER_PATH)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model = model.to("cuda:0")
model.eval()

print(
    "model loaded:",
    round(time.time() - load_start, 2),
    "seconds"
)

print(
    "GPU memory:",
    round(
        torch.cuda.memory_allocated() / 1024**3,
        2
    ),
    "GB"
)


# ============================================================
# Resume
# ============================================================

completed = load_completed_tasks(RAW_PATH)

print("\n===== RESUME =====")
print("already completed:", len(completed))
print("remaining:", len(task_ids) - len(completed))


# ============================================================
# Generation
# ============================================================

total_start = time.time()

with open(RAW_PATH, "a", encoding="utf-8") as fout:

    for index, task_id in enumerate(task_ids):

        if task_id in completed:
            continue

        problem = problems[task_id]
        prompt = problem["prompt"]

        # Same task gets same seed across Base / M0 / M1 later.
        task_seed = BASE_SEED + index

        set_seed(task_seed)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda:0")

        start = time.time()

        with torch.inference_mode():

            output_ids = model.generate(
                **inputs,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=True,
                temperature=TEMPERATURE,

                # Explicitly disable additional sampling truncation.
                top_p=TOP_P,
                top_k=TOP_K,

                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start

        input_len = inputs["input_ids"].shape[1]

        new_ids = output_ids[0, input_len:]

        raw_completion = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        )

        cleaned_completion = truncate_completion(
            raw_completion
        )

        solution = prompt + cleaned_completion

        record = {
            "task_id": task_id,

            "prompt": prompt,

            "raw_completion": raw_completion,

            "cleaned_completion": cleaned_completion,

            # Full executable candidate for EvalPlus
            "solution": solution,

            "generation": {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "do_sample": True,
                "seed": task_seed,
            },

            "stats": {
                "generated_tokens": int(len(new_ids)),
                "generation_seconds": elapsed,
                "raw_chars": len(raw_completion),
                "cleaned_chars": len(cleaned_completion),
            },
        }

        fout.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )

        fout.flush()

        finished = index + 1

        print(
            f"[{finished:03d}/164] "
            f"{task_id:<15} "
            f"tokens={len(new_ids):3d} "
            f"time={elapsed:6.2f}s "
            f"raw={len(raw_completion):4d} "
            f"clean={len(cleaned_completion):4d}"
        )


# ============================================================
# Build EvalPlus file
# ============================================================

rebuild_eval_file(
    RAW_PATH,
    EVAL_PATH,
)


# ============================================================
# Summary
# ============================================================

elapsed_total = time.time() - total_start

completed = load_completed_tasks(RAW_PATH)

print("\n===== GENERATION COMPLETE =====")
print("completed tasks:", len(completed))
print("total generation seconds:", round(elapsed_total, 2))

print("\nRAW:")
print(RAW_PATH)

print("\nEVAL:")
print(EVAL_PATH)
