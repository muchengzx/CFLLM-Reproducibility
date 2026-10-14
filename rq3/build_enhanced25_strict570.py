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


DETAIL = WORK_DIR / 'rq3_enhanced25_python570_n4_contract.json'

PROMPTS = PROMPT_DIR / 'rq3_enhanced_python570.json'

TRAIN_OUT = WORK_DIR / 'rq3_enhanced25_python570_strict.json'

META_OUT = WORK_DIR / 'rq3_enhanced25_python570_strict_meta.json'


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


def normalize(code):
    return "\n".join(
        line.rstrip()
        for line in textwrap.dedent(
            strip_outer_fence(code)
        ).strip().splitlines()
    )


def nonempty_rows(code):
    return sum(
        1
        for line in normalize(code).splitlines()
        if line.strip()
    )


def candidate_key(r):
    ratio = r["char_ratio"]

    if ratio < 0.20 or ratio > 1.50:
        length_risk = 2

    elif ratio < 0.35 or ratio > 1.20:
        length_risk = 1

    else:
        length_risk = 0

    output = strip_outer_fence(
        r["output"]
    )

    source = r["input"]

    unchanged = (
        normalize(output)
        ==
        normalize(source)
    )

    # reliability first,
    # then prefer actual refactoring,
    # then shorter code
    return (
        length_risk,
        int(unchanged),
        nonempty_rows(output),
        r["candidate_id"],
    )


rows = json.loads(
    DETAIL.read_text(
        encoding="utf-8"
    )
)

prompts = json.loads(
    PROMPTS.read_text(
        encoding="utf-8"
    )
)

prompt_map = {
    x["id"]: x
    for x in prompts
}

G = defaultdict(list)

for r in rows:
    G[r["id"]].append(r)


assert len(G) == 570


train = []
meta = []

stats = Counter()


for pid in sorted(
    prompt_map,
    key=lambda x: int(
        x.split("_")[-1]
    )
):

    prompt = prompt_map[pid]

    # STRICT:
    # contract + footprint must both pass
    candidates = [
        r
        for r in G[pid]
        if (
            r["contract_pass"]
            and
            r["footprint_pass"]
        )
    ]

    if candidates:

        best = min(
            candidates,
            key=candidate_key
        )

        output = strip_outer_fence(
            best["output"]
        )

        source = prompt["input"]

        same = (
            normalize(output)
            ==
            normalize(source)
        )

        train.append({
            "instruction": prompt["instruction"],
            "input": source,
            "output": output,
        })

        meta.append({
            "id": pid,
            "source": "strict_candidate",
            "candidate_id": best["candidate_id"],
            "available_strict_candidates":
                len(candidates),
            "char_ratio": best["char_ratio"],
            "text_same": same,
            "source_rows":
                nonempty_rows(source),
            "selected_rows":
                nonempty_rows(output),
        })

        stats["candidate"] += 1

        if same:
            stats["same"] += 1

        if best["char_ratio"] < 0.20:
            stats["lt020"] += 1

        if best["char_ratio"] < 0.35:
            stats["lt035"] += 1

    else:

        source = prompt["input"]

        train.append({
            "instruction": prompt["instruction"],
            "input": source,
            "output": source,
        })

        meta.append({
            "id": pid,
            "source": "original_anchor",
            "candidate_id": None,
            "available_strict_candidates": 0,
            "char_ratio": 1.0,
            "text_same": True,
            "source_rows":
                nonempty_rows(source),
            "selected_rows":
                nonempty_rows(source),
        })

        stats["anchor"] += 1


assert len(train) == 570
assert len(meta) == 570


TRAIN_OUT.write_text(
    json.dumps(
        train,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

META_OUT.write_text(
    json.dumps(
        meta,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


source_rows = sum(
    x["source_rows"]
    for x in meta
)

selected_rows = sum(
    x["selected_rows"]
    for x in meta
)


print("=" * 80)
print("STRICT ENHANCED RQ3 — PYTHON570")
print("=" * 80)

print(
    "Total targets            :",
    len(train)
)

print(
    "Strict clean candidates  :",
    stats["candidate"]
)

print(
    "Original safe anchors    :",
    stats["anchor"]
)

print(
    "Candidate unchanged      :",
    stats["same"]
)

print(
    "Candidate changed        :",
    stats["candidate"]
    - stats["same"]
)

print(
    "Selected ratio < 0.20    :",
    stats["lt020"]
)

print(
    "Selected ratio < 0.35    :",
    stats["lt035"]
)

print()
print(
    "Total source rows        :",
    source_rows
)

print(
    "Total selected rows      :",
    selected_rows
)

print(
    "Row retention            :",
    f"{selected_rows/source_rows:.3f}"
)

print(
    "Row reduction            :",
    f"{100*(1-selected_rows/source_rows):.2f}%"
)

print()
print("Saved training set:", TRAIN_OUT)
print("Saved metadata    :", META_OUT)
