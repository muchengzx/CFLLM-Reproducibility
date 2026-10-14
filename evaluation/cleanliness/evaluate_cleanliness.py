import ast
import json
import argparse
import statistics
from pathlib import Path

from radon.complexity import cc_visit
from evalplus.data import get_human_eval_plus


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_summary", required=True)
    return p.parse_args()


def count_args(node):
    args = node.args

    n = (
        len(args.posonlyargs)
        + len(args.args)
        + len(args.kwonlyargs)
    )

    if args.vararg is not None:
        n += 1

    if args.kwarg is not None:
        n += 1

    return n


def find_target_function(tree, entry_point):
    # HumanEval target is a top-level function.
    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == entry_point:
            return node

    return None


def find_cc(full_code, entry_point, target_lineno):
    blocks = cc_visit(full_code)

    candidates = [
        b for b in blocks
        if getattr(b, "name", None) == entry_point
    ]

    if not candidates:
        return None

    # Prefer the radon block corresponding to the same AST function line.
    for block in candidates:
        if getattr(block, "lineno", None) == target_lineno:
            return block.complexity

    # HumanEval normally only has one matching top-level function.
    return candidates[0].complexity


def make_stats(values):
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    problems = get_human_eval_plus()

    cc_values = []
    row_values = []
    arg_values = []

    total = 0
    valid = 0
    invalid = 0

    invalid_tasks = []

    with input_path.open("r", encoding="utf-8") as fin, \
         output_jsonl.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)

            task_id = record["task_id"]

            result = {
                "task_id": task_id,
                "valid": False,
                "cc": None,
                "rows": None,
                "args": None,
                "error": None,
            }

            try:
                if task_id not in problems:
                    raise ValueError(
                        f"task not found in EvalPlus: {task_id}"
                    )

                problem = problems[task_id]
                entry_point = problem["entry_point"]

                prompt = record["prompt"]

                if "cleaned_completion" not in record:
                    raise KeyError(
                        "raw record has no cleaned_completion"
                    )

                cleaned_completion = record["cleaned_completion"]

                # Frozen protocol:
                # evaluate exactly prompt + cleaned generated body.
                full_code = prompt + cleaned_completion

                tree = ast.parse(full_code)

                target = find_target_function(
                    tree,
                    entry_point,
                )

                if target is None:
                    raise ValueError(
                        f"target function not found: {entry_point}"
                    )

                if target.end_lineno is None:
                    raise ValueError(
                        "AST node has no end_lineno"
                    )

                rows = (
                    target.end_lineno
                    - target.lineno
                    + 1
                )

                args_count = count_args(target)

                cc = find_cc(
                    full_code,
                    entry_point,
                    target.lineno,
                )

                if cc is None:
                    raise ValueError(
                        f"radon CC block not found: {entry_point}"
                    )

                result.update({
                    "entry_point": entry_point,
                    "valid": True,
                    "cc": cc,
                    "rows": rows,
                    "args": args_count,
                })

                cc_values.append(cc)
                row_values.append(rows)
                arg_values.append(args_count)

                valid += 1

            except Exception as e:
                invalid += 1

                result["error"] = (
                    f"{type(e).__name__}: {e}"
                )

                invalid_tasks.append({
                    "task_id": task_id,
                    "error": result["error"],
                })

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "input": str(input_path),
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "cc": make_stats(cc_values),
        "rows": make_stats(row_values),
        "args": make_stats(arg_values),
        "invalid_tasks": invalid_tasks,
        "protocol": {
            "code": "prompt + cleaned_completion",
            "repair": False,
            "rows": "end_lineno - lineno + 1",
            "args": (
                "posonlyargs + args + kwonlyargs "
                "+ vararg + kwarg"
            ),
            "cc": (
                "radon.complexity.cc_visit; "
                "target block matched by entry_point"
            ),
            "aggregation": "valid generations only",
        },
    }

    with output_summary.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("===== CLEANLINESS =====")
    print("Total  :", total)
    print("Valid  :", valid)
    print("Invalid:", invalid)

    if valid:
        print()
        print(
            "CC mean   =",
            round(summary["cc"]["mean"], 4)
        )
        print(
            "Rows mean =",
            round(summary["rows"]["mean"], 4)
        )
        print(
            "Args mean =",
            round(summary["args"]["mean"], 4)
        )

        print()
        print("CC   :", summary["cc"])
        print("Rows :", summary["rows"])
        print("Args :", summary["args"])

    if invalid_tasks:
        print("\nInvalid tasks:")
        for x in invalid_tasks:
            print(
                " ",
                x["task_id"],
                "->",
                x["error"],
            )

    print("\nSaved:")
    print(output_jsonl)
    print(output_summary)


if __name__ == "__main__":
    main()
