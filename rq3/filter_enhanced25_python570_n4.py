import ast
import builtins
import json
import re
import statistics
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

from collections import Counter, defaultdict


INPUT = CANDIDATES_DIR / 'rq3_enhanced25_python570_n4.jsonl'

PROMPTS = PROMPT_DIR / 'rq3_enhanced_python570.json'

DETAIL_OUT = WORK_DIR / 'rq3_enhanced25_python570_n4_contract.json'

FAILED_PROMPTS_OUT = WORK_DIR / 'rq3_enhanced_python570_need_extra4.json'


BUILTINS = set(dir(builtins))

DANGEROUS_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "exit",
    "quit",
}

DANGEROUS_ROOTS = {
    "os",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "shutil",
    "pathlib",
    "multiprocessing",
    "threading",
}


def strip_outer_fence(code):
    if not code:
        return ""

    s = code.strip()

    m = re.fullmatch(
        r"```(?:python|py)?\s*\n?(.*?)\n?```",
        s,
        flags=re.I | re.S,
    )

    if m:
        return m.group(1).strip()

    return s


def normalize_text(code):
    return "\n".join(
        line.rstrip()
        for line in textwrap.dedent(
            strip_outer_fence(code)
        ).strip().splitlines()
    )


def parse(code):
    try:
        return ast.parse(
            textwrap.dedent(
                strip_outer_fence(code)
            )
        )
    except Exception:
        return None


def functions_in_order(tree):
    if tree is None:
        return []

    funcs = [
        n
        for n in ast.walk(tree)
        if isinstance(
            n,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            )
        )
    ]

    return sorted(
        funcs,
        key=lambda n: (
            getattr(n, "lineno", 10**9),
            getattr(n, "col_offset", 10**9),
        )
    )


def get_target_pair(source, candidate):
    st = parse(source)
    ct = parse(candidate)

    sf = functions_in_order(st)
    cf = functions_in_order(ct)

    if not sf:
        return st, ct, None, None

    src_fn = sf[0]
    cand_fn = None

    for f in cf:
        if f.name == src_fn.name:
            cand_fn = f
            break

    return st, ct, src_fn, cand_fn


def expr_repr(x):
    if x is None:
        return None

    return ast.dump(
        x,
        annotate_fields=True,
        include_attributes=False,
    )


def signature_repr(fn):
    if fn is None:
        return None

    a = fn.args

    return {
        "posonly": [
            x.arg for x in a.posonlyargs
        ],
        "args": [
            x.arg for x in a.args
        ],
        "vararg": (
            a.vararg.arg
            if a.vararg
            else None
        ),
        "kwonly": [
            x.arg for x in a.kwonlyargs
        ],
        "kwarg": (
            a.kwarg.arg
            if a.kwarg
            else None
        ),
        "defaults": [
            expr_repr(x)
            for x in a.defaults
        ],
        "kw_defaults": [
            expr_repr(x)
            for x in a.kw_defaults
        ],
    }


def iter_function_nodes(fn):
    if fn is None:
        return

    stack = list(fn.body)

    while stack:
        node = stack.pop()

        yield node

        # Do not let nested definitions pollute
        # the target function's behavior profile.
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.ClassDef,
            )
        ):
            continue

        stack.extend(
            ast.iter_child_nodes(node)
        )


def generator_kind(fn):
    if fn is None:
        return None

    return any(
        isinstance(
            n,
            (
                ast.Yield,
                ast.YieldFrom,
            )
        )
        for n in iter_function_nodes(fn)
    )


def dotted_name(node):
    parts = []
    cur = node

    while isinstance(
        cur,
        ast.Attribute
    ):
        parts.append(cur.attr)
        cur = cur.value

    if isinstance(cur, ast.Name):
        parts.append(cur.id)

        return ".".join(
            reversed(parts)
        )

    return None


def decorator_set(fn):
    if fn is None:
        return set()

    result = set()

    for d in fn.decorator_list:

        if isinstance(d, ast.Name):
            result.add(d.id)

        elif isinstance(d, ast.Attribute):
            name = dotted_name(d)

            if name:
                result.add(name)

        elif isinstance(d, ast.Call):
            name = dotted_name(d.func)

            if name:
                result.add(name)

    return result


def self_attr_writes(fn):
    if fn is None:
        return set()

    result = set()

    for n in iter_function_nodes(fn):

        if (
            isinstance(n, ast.Attribute)
            and
            isinstance(n.ctx, ast.Store)
        ):
            root = n.value

            if (
                isinstance(root, ast.Name)
                and
                root.id in {"self", "cls"}
            ):
                result.add(
                    f"{root.id}.{n.attr}"
                )

    return result


def global_nonlocal(fn):
    if fn is None:
        return set()

    result = set()

    for n in iter_function_nodes(fn):

        if isinstance(n, ast.Global):
            result.update(
                "global:" + x
                for x in n.names
            )

        elif isinstance(n, ast.Nonlocal):
            result.update(
                "nonlocal:" + x
                for x in n.names
            )

    return result


def dangerous_ops(fn):
    if fn is None:
        return set()

    result = set()

    for n in iter_function_nodes(fn):

        if not isinstance(n, ast.Call):
            continue

        f = n.func

        if isinstance(f, ast.Name):

            if f.id in DANGEROUS_NAMES:
                result.add(f.id)

        elif isinstance(f, ast.Attribute):

            name = dotted_name(f)

            if not name:
                continue

            root = name.split(".")[0]

            if root in DANGEROUS_ROOTS:
                result.add(name)

    return result


def local_names(fn):
    if fn is None:
        return set()

    result = set()

    for arg in (
        list(fn.args.posonlyargs)
        + list(fn.args.args)
        + list(fn.args.kwonlyargs)
    ):
        result.add(arg.arg)

    if fn.args.vararg:
        result.add(
            fn.args.vararg.arg
        )

    if fn.args.kwarg:
        result.add(
            fn.args.kwarg.arg
        )

    for n in iter_function_nodes(fn):

        if (
            isinstance(n, ast.Name)
            and
            isinstance(n.ctx, ast.Store)
        ):
            result.add(n.id)

        if isinstance(
            n,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            )
        ):
            result.add(n.name)

        if isinstance(n, ast.Import):
            for alias in n.names:
                result.add(
                    alias.asname
                    or alias.name.split(".")[0]
                )

        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                result.add(
                    alias.asname
                    or alias.name
                )

    return result


def free_names(fn):
    if fn is None:
        return set()

    locals_ = local_names(fn)
    result = set()

    for n in iter_function_nodes(fn):

        if not (
            isinstance(n, ast.Name)
            and
            isinstance(n.ctx, ast.Load)
        ):
            continue

        if n.id in locals_:
            continue

        if n.id in BUILTINS:
            continue

        result.add(n.id)

    return result


def external_calls(fn):
    if fn is None:
        return set()

    locals_ = local_names(fn)
    result = set()

    for n in iter_function_nodes(fn):

        if not isinstance(n, ast.Call):
            continue

        name = dotted_name(n.func)

        if not name:
            continue

        root = name.split(".")[0]

        if root in locals_:
            continue

        if root in BUILTINS:
            continue

        result.add(name)

    return result


def analyze(x):
    source = x["input"]
    output = x["output"]

    st, ct, sf, cf = get_target_pair(
        source,
        output,
    )

    parse_ok = (
        ct is not None
    )

    target_ok = (
        cf is not None
    )

    sig_ok = (
        sf is not None
        and
        cf is not None
        and
        signature_repr(sf)
        ==
        signature_repr(cf)
    )

    async_same = (
        sf is not None
        and
        cf is not None
        and
        isinstance(
            sf,
            ast.AsyncFunctionDef
        )
        ==
        isinstance(
            cf,
            ast.AsyncFunctionDef
        )
    )

    generator_same = (
        sf is not None
        and
        cf is not None
        and
        generator_kind(sf)
        ==
        generator_kind(cf)
    )

    decorators_same = (
        sf is not None
        and
        cf is not None
        and
        decorator_set(sf)
        ==
        decorator_set(cf)
    )

    self_writes_same = (
        sf is not None
        and
        cf is not None
        and
        self_attr_writes(sf)
        ==
        self_attr_writes(cf)
    )

    global_same = (
        sf is not None
        and
        cf is not None
        and
        global_nonlocal(sf)
        ==
        global_nonlocal(cf)
    )

    dangerous_same = (
        sf is not None
        and
        cf is not None
        and
        dangerous_ops(sf)
        ==
        dangerous_ops(cf)
    )

    contract_pass = all([
        parse_ok,
        target_ok,
        sig_ok,
        async_same,
        generator_same,
        decorators_same,
        self_writes_same,
        global_same,
        dangerous_same,
    ])

    free_same = (
        sf is not None
        and
        cf is not None
        and
        free_names(sf)
        ==
        free_names(cf)
    )

    calls_same = (
        sf is not None
        and
        cf is not None
        and
        external_calls(sf)
        ==
        external_calls(cf)
    )

    footprint_pass = (
        contract_pass
        and free_same
        and calls_same
    )

    src_len = len(source)
    out_len = len(output)

    char_ratio = (
        out_len / src_len
        if src_len
        else 0.0
    )

    return {
        "id": x["id"],
        "candidate_id": int(
            x["candidate_id"]
        ),
        "parse_ok": parse_ok,
        "target_ok": target_ok,
        "signature_same": sig_ok,
        "async_same": async_same,
        "generator_same": generator_same,
        "decorators_same": decorators_same,
        "self_writes_same": self_writes_same,
        "global_nonlocal_same": global_same,
        "dangerous_same": dangerous_same,
        "contract_pass": contract_pass,
        "free_names_same": free_same,
        "external_calls_same": calls_same,
        "footprint_pass": footprint_pass,
        "char_ratio": char_ratio,
        "text_same": (
            normalize_text(source)
            ==
            normalize_text(output)
        ),
        "input": source,
        "output": output,
    }


rows = []

with INPUT.open(
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        rows.append(
            analyze(
                json.loads(line)
            )
        )


assert len(rows) == 2280


G = defaultdict(list)

for r in rows:
    G[r["id"]].append(r)


assert len(G) == 570


def pct(n, d):
    return (
        100.0 * n / d
        if d
        else 0.0
    )


print("=" * 82)
print("ENHANCED PYTHON570 — n=4 CONTRACT FILTER")
print("=" * 82)

metrics = [
    ("parse_ok", "AST parseable"),
    ("target_ok", "Target preserved"),
    ("signature_same", "Exact signature"),
    ("contract_pass", "CONTRACT PASS"),
    ("footprint_pass", "FOOTPRINT PASS"),
    ("text_same", "Exact text copy"),
]

for key, label in metrics:

    n = sum(
        r[key]
        for r in rows
    )

    print(
        f"{label:24s}: "
        f"{n:4d}/2280 "
        f"({pct(n,2280):6.2f}%)"
    )


print()
print("=" * 82)
print("PROMPT-LEVEL COVERAGE")
print("=" * 82)

signature_covered = 0
contract_covered = 0
footprint_covered = 0

distribution = Counter()

failed_ids = []


for pid, candidates in G.items():

    sig_n = sum(
        x["signature_same"]
        for x in candidates
    )

    contract_n = sum(
        x["contract_pass"]
        for x in candidates
    )

    footprint_n = sum(
        x["footprint_pass"]
        for x in candidates
    )

    if sig_n > 0:
        signature_covered += 1

    if contract_n > 0:
        contract_covered += 1
    else:
        failed_ids.append(pid)

    if footprint_n > 0:
        footprint_covered += 1

    distribution[contract_n] += 1


print(
    ">=1 exact-signature candidate:",
    f"{signature_covered}/570",
    f"({pct(signature_covered,570):.2f}%)"
)

print(
    ">=1 contract-pass candidate   :",
    f"{contract_covered}/570",
    f"({pct(contract_covered,570):.2f}%)"
)

print(
    ">=1 footprint-pass candidate  :",
    f"{footprint_covered}/570",
    f"({pct(footprint_covered,570):.2f}%)"
)

print(
    "Need extra candidates         :",
    f"{len(failed_ids)}/570",
    f"({pct(len(failed_ids),570):.2f}%)"
)


print()
print("Contract-pass distribution:")

for k in range(5):
    print(
        f"  {k}/4 : "
        f"{distribution[k]} prompts"
    )


contract_rows = [
    r
    for r in rows
    if r["contract_pass"]
]

ratios = [
    r["char_ratio"]
    for r in contract_rows
]


print()
print("=" * 82)
print("CONTRACT-PASS CANDIDATE CHARACTERISTICS")
print("=" * 82)

print(
    "Candidates:",
    len(contract_rows)
)

if ratios:

    print(
        "Mean char ratio   :",
        f"{statistics.mean(ratios):.3f}"
    )

    print(
        "Median char ratio :",
        f"{statistics.median(ratios):.3f}"
    )

    lt020 = sum(
        x < 0.20
        for x in ratios
    )

    lt035 = sum(
        x < 0.35
        for x in ratios
    )

    gt120 = sum(
        x > 1.20
        for x in ratios
    )

    print(
        "ratio < 0.20      :",
        f"{lt020}/{len(ratios)}",
        f"({pct(lt020,len(ratios)):.2f}%)"
    )

    print(
        "ratio < 0.35      :",
        f"{lt035}/{len(ratios)}",
        f"({pct(lt035,len(ratios)):.2f}%)"
    )

    print(
        "ratio > 1.20      :",
        f"{gt120}/{len(ratios)}",
        f"({pct(gt120,len(ratios)):.2f}%)"
    )


prompt_data = json.loads(
    PROMPTS.read_text(
        encoding="utf-8"
    )
)

prompt_map = {
    x["id"]: x
    for x in prompt_data
}

failed_ids = sorted(
    failed_ids,
    key=lambda x: int(
        x.split("_")[-1]
    )
)

failed_prompts = [
    prompt_map[pid]
    for pid in failed_ids
]


DETAIL_OUT.write_text(
    json.dumps(
        rows,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8"
)


FAILED_PROMPTS_OUT.write_text(
    json.dumps(
        failed_prompts,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8"
)


print()
print("=" * 82)
print("OUTPUT")
print("=" * 82)

print(
    "Detail:",
    DETAIL_OUT
)

print(
    "Need extra4:",
    FAILED_PROMPTS_OUT
)

print(
    "Failed prompt IDs:",
    failed_ids
)
