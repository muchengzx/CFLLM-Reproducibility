import argparse
import ast
import json
from pathlib import Path

from bigcodebench.generate import sanitize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def has_entry_function(code, entry_point):
    try:
        tree = ast.parse(code)
    except Exception:
        return False

    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if node.name == entry_point:
                return True

    return False


def main():
    args = parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    rows = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 1140

    ast_ok = 0
    entry_ok = 0
    empty = []
    bad_ast = []
    missing_entry = []
    remaining_fence = []
    changed = 0

    with dst.open("w", encoding="utf-8") as f:

        for x in rows:

            raw = x["raw_generation"]

            solution = sanitize(
                raw,
                x["entry_point"],
            )

            if solution != raw:
                changed += 1

            if not solution.strip():
                empty.append(x["task_id"])

            try:
                ast.parse(solution)
                ast_ok += 1
            except Exception as e:
                bad_ast.append(
                    (x["task_id"], repr(e))
                )

            if has_entry_function(
                solution,
                x["entry_point"],
            ):
                entry_ok += 1
            else:
                missing_entry.append(x["task_id"])

            if "```" in solution:
                remaining_fence.append(x["task_id"])

            out = {
                "task_id": x["task_id"],
                "entry_point": x["entry_point"],
                "source_solution": x["source_solution"],
                "raw_generation": raw,
                "solution": solution,
            }

            f.write(
                json.dumps(out, ensure_ascii=False)
                + "\n"
            )

    print("=" * 72)
    print("samples             :", len(rows))
    print("changed by sanitizer:", changed)
    print("AST OK              :", ast_ok)
    print("AST failed          :", len(bad_ast))
    print("entry function OK   :", entry_ok)
    print("missing entry       :", len(missing_entry))
    print("empty               :", len(empty))
    print("remaining ```       :", len(remaining_fence))

    print("\nfirst bad AST:")
    for x in bad_ast[:20]:
        print(x)

    print("\nmissing entry ids:")
    print(missing_entry[:20])

    print("\nempty ids:")
    print(empty[:20])

    print("\nremaining fence ids:")
    print(remaining_fence[:20])

    print("\nsaved:")
    print(dst)


if __name__ == "__main__":
    main()
