import ast
import json
from pathlib import Path

from bigcodebench.generate import sanitize


SRC = Path(
    "/home/zhangx/CFLLM/code_clean/bigcodebench/"
    "baseline_qwen25coder7b_instruct_complete_td06.jsonl"
)

OUT = Path(
    "/home/zhangx/CFLLM/code_clean/bigcodebench/"
    "baseline_qwen25coder7b_instruct_complete_td06_sanitized.jsonl"
)


rows = [
    json.loads(line)
    for line in SRC.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

assert len(rows) == 1140
assert len({x["task_id"] for x in rows}) == 1140


n_ast_ok = 0
n_fence = 0
n_changed = 0
empty = []
bad_ast = []


with OUT.open("w", encoding="utf-8") as f:
    for x in rows:

        raw_solution = x["solution"]

        solution = sanitize(
            raw_solution,
            x["entry_point"],
        )

        if solution != raw_solution:
            n_changed += 1

        if "```" in solution:
            n_fence += 1

        if not solution.strip():
            empty.append(x["task_id"])

        try:
            ast.parse(solution)
            n_ast_ok += 1
        except Exception as e:
            bad_ast.append(
                (x["task_id"], repr(e))
            )

        out = {
            "task_id": x["task_id"],
            "solution": solution,
            "raw_solution": raw_solution,
            "entry_point": x["entry_point"],
        }

        f.write(
            json.dumps(out, ensure_ascii=False)
            + "\n"
        )


print("=" * 72)
print("samples       :", len(rows))
print("changed       :", n_changed)
print("AST OK        :", n_ast_ok)
print("AST failed    :", len(bad_ast))
print("remaining ``` :", n_fence)
print("empty         :", len(empty))

print("\nfirst bad AST:")
for x in bad_ast[:20]:
    print(x)

print("\nempty ids:")
print(empty[:20])

print("\nsaved:")
print(OUT)
