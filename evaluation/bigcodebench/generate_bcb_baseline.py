import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams


MODEL = "/mnt/d/CFLLM/models/Qwen2.5-Coder-7B-Instruct"

DATA = Path(
    "/home/zhangx/CFLLM/code_clean/bigcodebench/"
    "bigcodebench_v0.1.4.parquet"
)

OUT = Path(
    "/home/zhangx/CFLLM/code_clean/bigcodebench/"
    "baseline_qwen25coder7b_instruct_complete_td06.jsonl"
)


def main():

    print("=" * 80)
    print("BigCodeBench baseline generation")
    print("MODEL:", MODEL)
    print("DATA :", DATA)
    print("OUT  :", OUT)
    print("=" * 80)

    df = pd.read_parquet(DATA)

    assert len(df) == 1140, f"Expected 1140 tasks, got {len(df)}"
    assert df["task_id"].nunique() == 1140

    prompts = df["complete_prompt"].tolist()

    sampling = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=4096,
        n=1,
        seed=20261005,
    )

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )

    outputs = llm.generate(
        prompts,
        sampling,
        use_tqdm=True,
    )

    assert len(outputs) == len(df)

    with OUT.open("w", encoding="utf-8") as f:

        for row, out in zip(df.to_dict("records"), outputs):

            completion = out.outputs[0].text

            # BigCodeBench Complete:
            # complete_prompt already contains the imports,
            # function signature and docstring.
            solution = row["complete_prompt"] + completion

            item = {
                "task_id": row["task_id"],
                "completion": completion,
                "solution": solution,
                "entry_point": row["entry_point"],
            }

            f.write(
                json.dumps(item, ensure_ascii=False)
                + "\n"
            )

    print("\n" + "=" * 80)
    print("saved  :", OUT)
    print("samples:", len(df))
    print("=" * 80)

    # Quick inspection
    with OUT.open("r", encoding="utf-8") as f:
        for i in range(3):
            x = json.loads(next(f))

            print("\nTASK:", x["task_id"])
            print("completion chars:", len(x["completion"]))
            print(x["completion"][:1200])
            print("-" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    main()
