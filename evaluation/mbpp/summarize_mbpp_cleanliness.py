import ast
import json
from pathlib import Path
from statistics import mean

from radon.complexity import cc_visit


ROOT = Path("/home/zhangx/CFLLM/code_clean")

FILES = {
    "RQ2-25": ROOT / "mbpp_rq2_25.jsonl",
    "RQ3-V1": ROOT / "mbpp_rq3_25_strict1570.jsonl",
}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def nonempty_rows(code):
    return sum(
        1
        for line in code.splitlines()
        if line.strip()
    )


def max_args(code):
    tree = ast.parse(code)

    values = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            # 保持和之前 HumanEval 的口径一致：
            # 只统计 positional args，即 len(args.args)
            values.append(
                len(node.args.args)
            )

    return max(values) if values else 0


def max_cc(code):
    try:
        blocks = cc_visit(code)

        if not blocks:
            return 1

        return max(
            block.complexity
            for block in blocks
        )

    except Exception:
        return None


def evaluate(path):
    rows = load_jsonl(path)

    cc_values = []
    arg_values = []
    row_values = []

    ast_ok = 0
    cc_ok = 0

    bad_ast = []
    bad_cc = []

    for item in rows:
        task_id = item["task_id"]
        code = item.get("completion", "")

        row_values.append(
            nonempty_rows(code)
        )

        try:
            arg_values.append(
                max_args(code)
            )
            ast_ok += 1
        except Exception as e:
            bad_ast.append(
                (task_id, repr(e))
            )

        cc = max_cc(code)

        if cc is not None:
            cc_values.append(cc)
            cc_ok += 1
        else:
            bad_cc.append(task_id)

    return {
        "n": len(rows),

        "CC": (
            mean(cc_values)
            if cc_values else float("nan")
        ),

        "Args": (
            mean(arg_values)
            if arg_values else float("nan")
        ),

        "Rows": mean(row_values),

        "AST_OK": ast_ok,
        "CC_OK": cc_ok,

        "bad_ast": bad_ast,
        "bad_cc": bad_cc,
    }


results = {}

print("=" * 80)
print("MBPP CLEANLINESS")
print("=" * 80)

for name, path in FILES.items():

    r = evaluate(path)
    results[name] = r

    print()
    print(name)
    print("-" * 60)

    print("Samples :", r["n"])
    print("CC      :", f'{r["CC"]:.4f}')
    print("Args    :", f'{r["Args"]:.4f}')
    print("Rows    :", f'{r["Rows"]:.4f}')
    print(
        "AST OK  :",
        f'{r["AST_OK"]}/{r["n"]}'
    )
    print(
        "CC OK   :",
        f'{r["CC_OK"]}/{r["n"]}'
    )

    if r["bad_ast"]:
        print(
            "Bad AST :",
            r["bad_ast"][:10]
        )


r2 = results["RQ2-25"]
r3 = results["RQ3-V1"]

print()
print("=" * 80)
print("RQ3 - RQ2")
print("=" * 80)

for metric in ["CC", "Args", "Rows"]:

    delta = (
        r3[metric]
        - r2[metric]
    )

    pct = (
        delta / r2[metric] * 100
        if r2[metric] != 0
        else float("nan")
    )

    direction = (
        "better"
        if delta < 0
        else "worse"
        if delta > 0
        else "same"
    )

    print(
        f"{metric:5s}: "
        f"{delta:+.4f} "
        f"({pct:+.2f}%) "
        f"[{direction}]"
    )
