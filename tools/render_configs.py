import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
templates = root / "configs"
destination = root / "runtime_configs"

dataset = os.environ.get("CFLLM_DATASET_DIR")
if not dataset:
    raise SystemExit("Set CFLLM_DATASET_DIR first.")

output_root = root / "workspace" / "outputs"

values = {
    "__CFLLM_BASE_MODEL__":
        os.environ.get(
            "CFLLM_BASE_MODEL",
            "Qwen/Qwen2.5-Coder-7B-Instruct"
        ),
    "__CFLLM_DATASET_DIR__": dataset,
    "__CFLLM_RQ2_OUTPUT__":
        str(output_root / "sft_QWcodemix25"),
    "__CFLLM_RQ3_OUTPUT__":
        str(output_root / "sft_QWcodemix25_rq3_enhanced_strict1570"),
}

values["__CFLLM_RQ2_ADAPTER__"] = os.environ.get(
    "CFLLM_RQ2_ADAPTER",
    values["__CFLLM_RQ2_OUTPUT__"]
)

values["__CFLLM_RQ3_ADAPTER__"] = os.environ.get(
    "CFLLM_RQ3_ADAPTER",
    values["__CFLLM_RQ3_OUTPUT__"]
)

destination.mkdir(parents=True, exist_ok=True)

for path in sorted(templates.glob("*.yaml.template")):
    content = path.read_text(encoding="utf-8")

    for key, value in values.items():
        content = content.replace(key, value)

    if "__CFLLM_" in content:
        raise ValueError(f"Unresolved variable: {path.name}")

    output_path = destination / path.name.removesuffix(".template")
    output_path.write_text(content, encoding="utf-8")
    print("[GENERATED]", output_path)

print("Configuration rendering: PASS")
