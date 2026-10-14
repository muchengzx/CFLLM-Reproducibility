import os
import json
import time
from pathlib import Path
from openai import OpenAI

DATA_ROOT = Path(os.environ.get('CFLLM_DATASET_DIR', str(Path(__file__).resolve().parents[1] / 'data')))
INPUT = Path(os.environ.get('CFLLM_RQ3_PY_INPUT', str(DATA_ROOT / 'rq3' / 'rq3_enhanced_python570.json')))

OUTPUT = Path(os.environ.get('CFLLM_RQ3_PY_OUTPUT', str(Path(__file__).resolve().parents[1] / 'workspace' / 'rq3' / 'rq3_enhanced25_python570_n4.jsonl')))

N_CANDIDATES = 4

client = OpenAI(
    api_key=os.environ.get("CFLLM_API_KEY", "EMPTY"),
    base_url=os.environ.get("CFLLM_API_BASE_URL", 'http://127.0.0.1:7000/v1'),
    timeout=600.0,
)


def build_content(item):
    return (
        item["instruction"].strip()
        + "\n\n"
        + item["input"].strip()
    )


data = json.loads(
    INPUT.read_text(
        encoding="utf-8"
    )
)

assert len(data) == 570


done = set()

if OUTPUT.exists():

    for line in OUTPUT.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines():

        if not line.strip():
            continue

        try:
            x = json.loads(line)

            done.add(
                (
                    x["id"],
                    int(x["candidate_id"])
                )
            )

        except Exception:
            pass


print("=" * 72)
print("ENHANCED RQ3 — PYTHON 570, n=4")
print("=" * 72)

print("Prompts:", len(data))
print("Candidates / prompt:", N_CANDIDATES)
print("Expected:", len(data) * N_CANDIDATES)
print("Already:", len(done))


try:
    client.models.list()
    print("API OK")
except Exception as e:
    print("API FAILED:", e)
    raise SystemExit(1)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open(
    "a",
    encoding="utf-8"
) as fout:

    for item in data:

        content = build_content(
            item
        )

        for cid in range(
            N_CANDIDATES
        ):

            key = (
                item["id"],
                cid
            )

            if key in done:
                continue

            success = False

            for attempt in range(3):

                try:

                    response = (
                        client.chat.completions.create(
                            model="default",
                            messages=[
                                {
                                    "role": "user",
                                    "content": content,
                                }
                            ],
                        )
                    )

                    output = (
                        response
                        .choices[0]
                        .message
                        .content
                        or ""
                    )

                    record = {
                        "id": item["id"],
                        "language": item["language"],
                        "candidate_id": cid,
                        "instruction": item["instruction"],
                        "input": item["input"],
                        "output": output,
                        "char_len": len(output),
                    }

                    fout.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                    fout.flush()

                    done.add(key)

                    print(
                        f"[{len(done):4d}/2280] "
                        f"{item['id']:12s} "
                        f"cand={cid} "
                        f"chars={len(output):5d}"
                    )

                    success = True
                    break

                except Exception as e:

                    print(
                        f"[WARN] "
                        f"{item['id']} "
                        f"cand={cid} "
                        f"attempt={attempt+1}: "
                        f"{e}"
                    )

                    time.sleep(5)

            if not success:
                print(
                    "[FAILED]",
                    item["id"],
                    cid
                )


print()
print("Finished.")
print("Generated:", len(done), "/2280")
print("Saved:", OUTPUT)
