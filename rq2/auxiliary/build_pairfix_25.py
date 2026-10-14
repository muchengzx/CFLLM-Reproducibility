import ast
import json
import re
import textwrap
from pathlib import Path
from collections import Counter

ROOT = Path("/home/zhangx/CFLLM")

SRC = {
    "py25": ROOT / "data" / "py25.json",
    "cpp25": ROOT / "data" / "cpp25.json",
}

OUT_DIR = ROOT / "data_pairfix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Python
# ---------------------------------------------------------
def get_python_func_name(code: str):
    """
    High-precision diagnostic parser:
    dedent snippet, parse AST, return the earliest function/method definition.
    """
    try:
        src = textwrap.dedent(code).strip()
        tree = ast.parse(src)

        funcs = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not funcs:
            return None

        funcs.sort(key=lambda n: (n.lineno, n.col_offset))
        return funcs[0].name

    except Exception:
        return None


# ---------------------------------------------------------
# C++
# ---------------------------------------------------------
CPP_BAD_NAMES = {
    "if", "for", "while", "switch", "catch",
    "sizeof", "alignof", "decltype",
    "return", "new", "delete",
}


def get_cpp_func_name(code: str):
    """
    Conservative heuristic:
    only inspect text before the first opening brace.
    This avoids the previous bug of matching function calls inside the body.

    It is intentionally conservative: unknown samples are rejected rather
    than guessed.
    """
    src = code.strip()
    if not src or "{" not in src:
        return None

    header = src.split("{", 1)[0]

    # remove comments in the header
    header = re.sub(r"/\*.*?\*/", " ", header, flags=re.S)
    header = re.sub(r"//.*?$", " ", header, flags=re.M)

    # Match qualified C++ names followed by "(".
    # Examples:
    #   foo(
    #   Class::foo(
    #   ns::Class::foo(
    #   Class::~Class(
    pattern = re.compile(
        r"(?P<name>"
        r"(?:[A-Za-z_~][A-Za-z0-9_~]*::)*"
        r"[A-Za-z_~][A-Za-z0-9_~]*"
        r")\s*\("
    )

    matches = list(pattern.finditer(header))
    if not matches:
        return None

    # In a normal function definition, the final plausible name before
    # the opening brace is usually the function being defined.
    for m in reversed(matches):
        name = m.group("name")
        leaf = name.split("::")[-1]

        if leaf not in CPP_BAD_NAMES:
            return name

    return None


def audit_and_filter(tag, path, lang):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    kept = []
    rejected = []

    same = 0
    different = 0
    unknown = 0

    for idx, item in enumerate(data):
        inp = item.get("input", "")
        out = item.get("output", "")

        if lang == "py":
            old_name = get_python_func_name(inp)
            new_name = get_python_func_name(out)
        else:
            old_name = get_cpp_func_name(inp)
            new_name = get_cpp_func_name(out)

        if old_name is None or new_name is None:
            unknown += 1
            rejected.append({
                "index": idx,
                "reason": "unknown",
                "old_name": old_name,
                "new_name": new_name,
                "item": item,
            })
            continue

        if old_name != new_name:
            different += 1
            rejected.append({
                "index": idx,
                "reason": "different_name",
                "old_name": old_name,
                "new_name": new_name,
                "item": item,
            })
            continue

        same += 1
        kept.append(item)

    # exact pair duplication statistics only; DO NOT remove yet
    pairs = [
        (x.get("input", "").strip(), x.get("output", "").strip())
        for x in kept
    ]

    duplicate_count = len(pairs) - len(set(pairs))

    print("\n" + "=" * 72)
    print(tag)
    print("original             :", len(data))
    print("kept same-name       :", same)
    print("rejected different   :", different)
    print("rejected unknown     :", unknown)
    print("duplicates among kept:", duplicate_count)
    print("keep rate            :", f"{same / len(data) * 100:.2f}%")

    print("\nFirst 10 rejected:")
    for x in rejected[:10]:
        print(
            x["index"],
            x["reason"],
            repr(x["old_name"]),
            "->",
            repr(x["new_name"]),
        )

    out_data = OUT_DIR / f"{tag}_pairfix.json"
    out_rejected = OUT_DIR / f"{tag}_rejected.json"

    with out_data.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    with out_rejected.open("w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(out_data)
    print(out_rejected)

    return {
        "dataset": tag,
        "original": len(data),
        "kept": same,
        "different": different,
        "unknown": unknown,
        "duplicates_among_kept": duplicate_count,
    }


summary = []

summary.append(
    audit_and_filter(
        "py25",
        SRC["py25"],
        "py",
    )
)

summary.append(
    audit_and_filter(
        "cpp25",
        SRC["cpp25"],
        "cpp",
    )
)

summary_path = OUT_DIR / "pairfix_25_summary.json"

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 72)
print("SUMMARY")
for x in summary:
    print(x)

print("\nSummary saved to:")
print(summary_path)
