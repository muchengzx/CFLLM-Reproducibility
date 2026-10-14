import json
import re
from pathlib import Path

from openai import OpenAI
from evalplus.data import get_mbpp_plus


OUT = Path(
    "/home/zhangx/CFLLM/code_clean/"
    "mbpp_baseline_qwen25coder7b_instruct_td06.jsonl"
)

SYSTEM_PROMPT = """You are an expert Python programmer.

Given a programming problem, write a complete and correct Python implementation.

Requirements:
- Implement the requested function exactly.
- Preserve the required function name.
- Include any imports needed by the implementation.
- Return only executable Python code.
- Do not include Markdown fences.
- Do not include explanations.
- Do not generate additional tests or a main function.
"""


def strip_fences(text):
    text = text or ""
    text = text.strip()

    text = re.sub(
        r"^```(?:python|Python)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def load_existing(path):
    if not path.exists():
        return []

    rows = []

    for line in path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines():

        if line.strip():
            rows.append(
                json.loads(line)
            )

    return rows


problems = get_mbpp_plus()

print("=" * 80)
print("MBPP BASELINE GENERATION")
print("=" * 80)
print("Tasks:", len(problems))
print("Output:", OUT)
print("=" * 80)


client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:7002/v1",
    timeout=600.0,
    max_retries=3,
)


existing = load_existing(OUT)

completed = {
    x["task_id"]
    for x in existing
}

print(
    "Resume mode:",
    len(completed),
    "already completed"
)


with OUT.open(
    "a",
    encoding="utf-8"
) as f:

    for idx, (task_id, problem) in enumerate(
        problems.items(),
        start=1,
    ):

        if task_id in completed:
            continue

        prompt = problem["prompt"]
        entry = problem["entry_point"]

        user_prompt = (
            prompt
            + "\n\n"
            + f"Required entry function: {entry}\n"
            + "Return the complete Python implementation only."
        )

        try:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.6,
                top_p=1.0,
                max_tokens=2048,
            )

            raw = (
                response.choices[0].message.content
                or ""
            )

            code = strip_fences(raw)

        except Exception as e:

            print(
                f"[ERROR] {task_id}: {e}",
                flush=True,
            )

            code = ""

        item = {
            "task_id": task_id,
            "completion": code,
        }

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()

        if idx % 10 == 0:
            print(
                f"{idx}/{len(problems)}",
                flush=True,
            )


print()
print("Finished.")
