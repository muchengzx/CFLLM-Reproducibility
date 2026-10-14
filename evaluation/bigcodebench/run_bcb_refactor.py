import argparse
import json
import multiprocessing as mp
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


BASE_MODEL = "/mnt/d/CFLLM/models/Qwen2.5-Coder-7B-Instruct"

INPUT_FILE = (
    "/home/zhangx/CFLLM/code_clean/bigcodebench/"
    "baseline_qwen25coder7b_instruct_complete_td06_sanitized.jsonl"
)


CLEAN_PROMPT = """Here is the original code. Please provide a cleaner, more concise, and more readable version by following clean code principles.
Import any necessary libraries and helper functions if required, but keep the definition of the entry function unchanged.

Your goals are:
- Reduce cyclomatic complexity where possible
- Decrease the number of lines of code without sacrificing clarity
- Keep 100% of the original functionality unchanged
- Do NOT add type hints or docstrings unless they already exist in the original code

Requirements:
- Return a complete executable Python module, not only a function body or a patch.
- Preserve the exact entry function name and parameter signature.
- Preserve all imports, global variables, constants, helper functions, classes, and dependencies that are necessary for the original behavior.
- Preserve return values, exceptions, state changes, randomness, I/O behavior, and other observable side effects.
- Do not add test code, demonstrations, Markdown fences, or explanations.
- Do not remove logic merely to make the implementation shorter.
- If no behavior-preserving simplification is clearly safe, return the original implementation unchanged.
- Return only the improved code.

Original code:
{code}
"""


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--adapter",
        required=True,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--name",
        required=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    src = Path(INPUT_FILE)
    out_path = Path(args.output)

    rows = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1140
    assert len({x["task_id"] for x in rows}) == 1140

    print("=" * 80)
    print("BigCodeBench refactoring")
    print("name    :", args.name)
    print("base    :", BASE_MODEL)
    print("adapter :", args.adapter)
    print("input   :", src)
    print("output  :", out_path)
    print("samples :", len(rows))
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    prompts = []

    for x in rows:

        user_text = CLEAN_PROMPT.format(
            code=x["solution"]
        )

        messages = [
            {
                "role": "user",
                "content": user_text,
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(prompt)

    sampling = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=4096,
        n=1,
        seed=20261005,
    )

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
    )

    lora_request = LoRARequest(
        args.name,
        1,
        args.adapter,
    )

    outputs = llm.generate(
        prompts,
        sampling,
        lora_request=lora_request,
        use_tqdm=True,
    )

    assert len(outputs) == len(rows)

    with out_path.open("w", encoding="utf-8") as f:

        for x, output in zip(rows, outputs):

            generation = output.outputs[0].text

            item = {
                "task_id": x["task_id"],
                "entry_point": x["entry_point"],
                "source_solution": x["solution"],
                "raw_generation": generation,
            }

            f.write(
                json.dumps(item, ensure_ascii=False)
                + "\n"
            )

    print("\n" + "=" * 80)
    print("saved  :", out_path)
    print("samples:", len(rows))
    print("=" * 80)

    # simple diagnostics
    empty = []
    fenced = []

    with out_path.open("r", encoding="utf-8") as f:
        saved = [json.loads(x) for x in f if x.strip()]

    for x in saved:
        if not x["raw_generation"].strip():
            empty.append(x["task_id"])
        if "```" in x["raw_generation"]:
            fenced.append(x["task_id"])

    print("empty generation:", len(empty))
    print("contains fences :", len(fenced))
    print("empty ids       :", empty[:20])
    print("fenced ids      :", fenced[:20])


if __name__ == "__main__":
    mp.freeze_support()
    main()
