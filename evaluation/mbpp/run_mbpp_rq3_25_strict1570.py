import ast
import json
import re
from pathlib import Path

from openai import OpenAI
from v4_postprocess import fix_code_v4


ROOT = Path("/home/zhangx/CFLLM/code_clean")

BASELINE = ROOT / "mbpp_baseline_qwen25coder7b_instruct_td06.jsonl"
V4_LOG = ROOT / "infclean_v4_log.txt"

RAW_OUT = ROOT / "mbpp_rq3_25_strict1570_raw.jsonl"
FINAL_OUT = ROOT / "mbpp_rq3_25_strict1570.jsonl"


OLD_INSTRUCTION = (
    "you must follow:\n"
    "    1. Provide the complete code without any textual explanations and do not generate test scripts.\n"
    "    2. Please strictly follow the specified format provided below for the code.\n"
    "    3. Do not change the function names.\n"
    "    4. The original code content must be fully included in the generated complete code, including all package import sections.\n"
    "    5. For Python language, do not generate test cases or main function.\n"
    "    6. Do not generate test cases.\n"
    "    "
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
        if line.strip()
    ]



V4_PROMPT_PREFIX = """Here is the original code. Please provide a cleaner, more concise, and more readable version by following clean code principles.
Import any necessary libraries and helper functions if required, but keep the definition of the entry function unchanged.

Your goals are:
- Reduce cyclomatic complexity where possible
- Decrease the number of lines of code without sacrificing clarity
- Keep 100% of the original functionality unchanged
- Do NOT add type hints or docstrings unless they already exist in the original code

Requirements:
- The output must define the function using `def` and preserve the original function name and parameter signature.
- The public interface must remain unchanged.
- You may import necessary libraries and define additional helper functions inside the implementation if needed.
- If the original implementation violates these requirements or contains obvious issues, fix it using reasonable judgment while preserving the intended functionality.

Return only the improved code.

"""


baseline = load_jsonl(BASELINE)

if len(baseline) != 378:
    raise RuntimeError(
        f"Expected 378 baseline samples, got {len(baseline)}"
    )

prompt_prefix = V4_PROMPT_PREFIX

print("=" * 80)
print("MBPP RQ3-25 Strict1570")
print("=" * 80)
print("Baseline samples :", len(baseline))
print("V4 prompt chars  :", len(prompt_prefix))
print("Raw output       :", RAW_OUT)
print("Final output     :", FINAL_OUT)
print("=" * 80)


client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:7000/v1",
    timeout=600.0
)


# Always start a fresh formal run.
RAW_OUT.unlink(missing_ok=True)
FINAL_OUT.unlink(missing_ok=True)


passed_request_count = 0

with (
    RAW_OUT.open("w", encoding="utf-8") as f_raw,
    FINAL_OUT.open("w", encoding="utf-8") as f_final
):

    for i, item in enumerate(baseline):

        task_id = item["task_id"]

        original_code = item["completion"].replace(
            OLD_INSTRUCTION,
            ""
        )

        user_content = prompt_prefix + original_code

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=0.6,
            top_p=1.0,
            max_tokens=4096
        )

        raw = (
            response.choices[0].message.content
            or ""
        )

        clean_code = fix_code_v4(raw)

        raw_item = {
            "task_id": task_id,
            "completion": raw
        }

        final_item = {
            "task_id": task_id,
            "completion": clean_code
        }

        f_raw.write(
            json.dumps(
                raw_item,
                ensure_ascii=False
            ) + "\n"
        )
        f_raw.flush()

        f_final.write(
            json.dumps(
                final_item,
                ensure_ascii=False
            ) + "\n"
        )
        f_final.flush()

        passed_request_count += 1

        raw_nonempty = sum(
            1 for line in raw.splitlines()
            if line.strip()
        )

        final_nonempty = sum(
            1 for line in clean_code.splitlines()
            if line.strip()
        )

        print(
            f"[{i + 1:3d}/164] "
            f"{task_id:15s} "
            f"raw_rows={raw_nonempty:3d} "
            f"final_rows={final_nonempty:3d}"
        )


print("\n" + "=" * 80)
print("FINISHED")
print("Generated :", passed_request_count)
print("Raw       :", RAW_OUT)
print("Final     :", FINAL_OUT)
print("=" * 80)
