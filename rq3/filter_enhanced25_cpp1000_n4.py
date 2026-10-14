import json
import re
import textwrap
from pathlib import Path


import os

RELEASE_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    os.environ.get(
        "CFLLM_DATASET_DIR",
        str(RELEASE_ROOT / "data")
    )
)

PROMPT_DIR = DATA_ROOT / "rq3"

CANDIDATES_DIR = Path(
    os.environ.get(
        "CFLLM_RQ3_CANDIDATES_DIR",
        str(DATA_ROOT)
    )
)

WORK_DIR = Path(
    os.environ.get(
        "CFLLM_RQ3_WORK_DIR",
        str(RELEASE_ROOT / "workspace" / "rq3")
    )
)

WORK_DIR.mkdir(parents=True, exist_ok=True)

from collections import defaultdict, Counter

from tree_sitter import Language, Parser
import tree_sitter_cpp


INPUT = CANDIDATES_DIR / 'rq3_enhanced25_cpp1000_n4.jsonl'

OUTPUT = WORK_DIR / 'rq3_enhanced25_cpp1000_n4_contract.json'


LANG = Language(tree_sitter_cpp.language())
PARSER = Parser(LANG)


def strip_outer_fence(code):
    if not code:
        return ""

    s = code.strip()

    m = re.fullmatch(
        r"```(?:cpp|c\+\+|cc|cxx)?\s*\n?(.*?)\n?```",
        s,
        flags=re.I | re.S,
    )

    if m:
        return m.group(1).strip()

    return s


def normalize(code):
    return "\n".join(
        line.rstrip()
        for line in textwrap.dedent(
            strip_outer_fence(code)
        ).strip().splitlines()
    )


def compact(s):
    # Remove comments and whitespace for
    # interface comparison.
    s = re.sub(
        r"/\*.*?\*/",
        "",
        s,
        flags=re.S,
    )
    s = re.sub(
        r"//[^\n]*",
        "",
        s,
    )
    return re.sub(r"\s+", "", s)


def node_text(node, code_bytes):
    return code_bytes[
        node.start_byte:node.end_byte
    ].decode(
        "utf-8",
        errors="ignore",
    )


def walk(node):
    yield node

    for child in node.named_children:
        yield from walk(child)


def parse_cpp(code):
    code = normalize(code)
    b = code.encode("utf-8")

    tree = PARSER.parse(b)

    return {
        "code": code,
        "bytes": b,
        "tree": tree,
        "ok": (
            bool(code.strip())
            and
            not tree.root_node.has_error
        ),
    }


def functions(parsed):
    return [
        n
        for n in walk(
            parsed["tree"].root_node
        )
        if n.type == "function_definition"
    ]


NAME_TYPES = {
    "identifier",
    "field_identifier",
    "operator_name",
    "destructor_name",
}


def function_name(fn, b):
    decl = fn.child_by_field_name(
        "declarator"
    )

    if decl is None:
        return None

    candidates = []

    def rec(n):
        if n.type == "parameter_list":
            return

        if n.type in NAME_TYPES:
            candidates.append(
                node_text(n, b)
            )

        for c in n.named_children:
            rec(c)

    rec(decl)

    if not candidates:
        return None

    # For A::B::func, the final identifier
    # before the parameter list is func.
    return candidates[-1]


def signature_text(fn, parsed):
    body = fn.child_by_field_name("body")

    if body is None:
        return ""

    start = fn.start_byte

    # Preserve template declaration if this
    # function is directly wrapped by one.
    parent = fn.parent

    if (
        parent is not None
        and
        parent.type == "template_declaration"
    ):
        start = parent.start_byte

    raw = parsed["bytes"][
        start:body.start_byte
    ].decode(
        "utf-8",
        errors="ignore",
    )

    return compact(raw)


def find_target(parsed, target_name):
    found = []

    for fn in functions(parsed):
        name = function_name(
            fn,
            parsed["bytes"],
        )

        if name == target_name:
            found.append(fn)

    return found


def callable_name(call, b):
    fn = call.child_by_field_name(
        "function"
    )

    if fn is None:
        return None

    return compact(
        node_text(fn, b)
    )


def call_footprint(fn, parsed):
    body = fn.child_by_field_name("body")

    if body is None:
        return set()

    result = set()

    for n in walk(body):
        if n.type == "call_expression":
            name = callable_name(
                n,
                parsed["bytes"],
            )

            if name:
                result.add(name)

    return result


def lhs_text(node, b):
    left = node.child_by_field_name(
        "left"
    )

    if left is None:
        left = node.child_by_field_name(
            "argument"
        )

    if left is None:
        return None

    return compact(
        node_text(left, b)
    )


def state_write_footprint(fn, parsed):
    body = fn.child_by_field_name("body")

    if body is None:
        return set()

    result = set()

    for n in walk(body):

        if n.type == "assignment_expression":
            lhs = lhs_text(
                n,
                parsed["bytes"],
            )

            if lhs and (
                "->" in lhs
                or "." in lhs
                or lhs.startswith("*")
            ):
                result.add(lhs)

        elif n.type == "update_expression":
            text = compact(
                node_text(
                    n,
                    parsed["bytes"],
                )
            )

            if (
                "->" in text
                or "." in text
                or "*" in text
            ):
                result.add(text)

    return result


RISK_TYPES = {
    "new_expression",
    "delete_expression",
    "throw_statement",
    "co_await_expression",
    "co_yield_statement",
    "goto_statement",
}


def risky_ops(fn):
    body = fn.child_by_field_name("body")

    if body is None:
        return Counter()

    return Counter(
        n.type
        for n in walk(body)
        if n.type in RISK_TYPES
    )


def analyze(source, output):
    src = parse_cpp(source)
    out = parse_cpp(output)

    result = {
        "source_parse": src["ok"],
        "parseable": out["ok"],
        "target_preserved": False,
        "exact_signature": False,
        "calls_same": False,
        "writes_same": False,
        "risky_ops_same": False,
        "contract_pass": False,
        "footprint_pass": False,
        "source_calls": [],
        "output_calls": [],
        "source_writes": [],
        "output_writes": [],
    }

    if not src["ok"]:
        return result

    src_funcs = functions(src)

    if not src_funcs:
        return result

    # Dataset instances are function-level.
    # Use the first source function as target.
    src_fn = src_funcs[0]

    target_name = function_name(
        src_fn,
        src["bytes"],
    )

    result["target_name"] = target_name

    if not out["ok"] or not target_name:
        return result

    candidates = find_target(
        out,
        target_name,
    )

    if not candidates:
        return result

    result["target_preserved"] = True

    # Prefer the candidate whose interface
    # matches the source exactly.
    src_sig = signature_text(
        src_fn,
        src,
    )

    exact = [
        fn
        for fn in candidates
        if signature_text(
            fn,
            out,
        ) == src_sig
    ]

    if not exact:
        return result

    out_fn = exact[0]

    result["exact_signature"] = True

    src_calls = call_footprint(
        src_fn,
        src,
    )

    out_calls = call_footprint(
        out_fn,
        out,
    )

    src_writes = state_write_footprint(
        src_fn,
        src,
    )

    out_writes = state_write_footprint(
        out_fn,
        out,
    )

    src_risky = risky_ops(src_fn)
    out_risky = risky_ops(out_fn)

    result["source_calls"] = sorted(
        src_calls
    )

    result["output_calls"] = sorted(
        out_calls
    )

    result["source_writes"] = sorted(
        src_writes
    )

    result["output_writes"] = sorted(
        out_writes
    )

    result["calls_same"] = (
        src_calls == out_calls
    )

    result["writes_same"] = (
        src_writes == out_writes
    )

    result["risky_ops_same"] = (
        src_risky == out_risky
    )

    # Contract = syntax + target + exact interface
    # + preserve explicit risky language constructs.
    result["contract_pass"] = all([
        result["parseable"],
        result["target_preserved"],
        result["exact_signature"],
        result["risky_ops_same"],
    ])

    # Strict footprint adds observable
    # external-call/state-write preservation.
    result["footprint_pass"] = all([
        result["contract_pass"],
        result["calls_same"],
        result["writes_same"],
    ])

    return result


records = []

with INPUT.open(
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        x = json.loads(line)

        source = x["input"]
        output = x["output"]

        a = analyze(
            source,
            output,
        )

        source_len = max(
            len(normalize(source)),
            1,
        )

        a.update({
            "id": x["id"],
            "language": x.get(
                "language",
                "cpp",
            ),
            "candidate_id":
                x["candidate_id"],
            "instruction":
                x["instruction"],
            "input": source,
            "output": output,
            "char_len":
                x.get(
                    "char_len",
                    len(output),
                ),
            "char_ratio":
                len(
                    normalize(output)
                ) / source_len,
            "text_same": (
                normalize(source)
                ==
                normalize(output)
            ),
        })

        records.append(a)


OUTPUT.write_text(
    json.dumps(
        records,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


N = len(records)


def count(key):
    return sum(
        bool(x.get(key))
        for x in records
    )


print("=" * 84)
print(
    "ENHANCED25 C++1000 — n=4 CONTRACT FILTER"
)
print("=" * 84)

for key, label in [
    ("parseable", "Parseable"),
    (
        "target_preserved",
        "Target preserved",
    ),
    (
        "exact_signature",
        "Exact signature",
    ),
    (
        "risky_ops_same",
        "Risky ops preserved",
    ),
    (
        "contract_pass",
        "CONTRACT PASS",
    ),
    (
        "calls_same",
        "Call footprint same",
    ),
    (
        "writes_same",
        "Write footprint same",
    ),
    (
        "footprint_pass",
        "FOOTPRINT PASS",
    ),
    (
        "text_same",
        "Exact text copy",
    ),
]:
    v = count(key)

    print(
        f"{label:24s}: "
        f"{v:4d}/{N} "
        f"({100*v/N:6.2f}%)"
    )


G = defaultdict(list)

for r in records:
    G[r["id"]].append(r)


print()
print("=" * 84)
print("PROMPT-LEVEL COVERAGE")
print("=" * 84)

for key, label in [
    (
        "exact_signature",
        ">=1 exact-signature candidate",
    ),
    (
        "contract_pass",
        ">=1 contract-pass candidate",
    ),
    (
        "footprint_pass",
        ">=1 footprint-pass candidate",
    ),
]:
    v = sum(
        any(x[key] for x in rows)
        for rows in G.values()
    )

    print(
        f"{label:34s}: "
        f"{v}/1000 "
        f"({100*v/1000:5.2f}%)"
    )


dist = Counter(
    sum(
        x["footprint_pass"]
        for x in rows
    )
    for rows in G.values()
)

print()
print("Strict footprint distribution:")

for i in range(5):
    print(
        f"  {i}/4 : "
        f"{dist[i]} prompts"
    )


passed = [
    r
    for r in records
    if r["footprint_pass"]
]

if passed:
    ratios = sorted(
        r["char_ratio"]
        for r in passed
    )

    print()
    print("=" * 84)
    print(
        "STRICT-PASS CANDIDATE CHARACTERISTICS"
    )
    print("=" * 84)

    print(
        "Candidates          :",
        len(passed),
    )

    print(
        "Mean char ratio     :",
        f"{sum(ratios)/len(ratios):.3f}",
    )

    print(
        "Median char ratio   :",
        f"{ratios[len(ratios)//2]:.3f}",
    )

    print(
        "ratio < 0.20        :",
        sum(x < 0.20 for x in ratios),
    )

    print(
        "ratio < 0.35        :",
        sum(x < 0.35 for x in ratios),
    )

    print(
        "ratio > 1.20        :",
        sum(x > 1.20 for x in ratios),
    )


print()
print("=" * 84)
print("OUTPUT")
print("=" * 84)
print("Detail:", OUTPUT)
