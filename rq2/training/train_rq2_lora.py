import os
import json
import math
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        type=str,
        default="/mnt/d/CFLLM/models/Qwen2.5-Coder-7B-Base",
    )

    p.add_argument("--py_data", type=str, required=True)
    p.add_argument("--cpp_data", type=str, required=True)

    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=4096)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # 用于 smoke test；0 表示完整训练
    p.add_argument("--max_steps", type=int, default=0)

    p.add_argument("--log_steps", type=int, default=10)

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_prompt(x):
    return (
        "### Instruction:\n"
        + x["instruction"].strip()
        + "\n\n### Input:\n"
        + x["input"].strip()
        + "\n\n### Response:\n"
    )


def load_one_file(path, source, tokenizer, max_length):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    valid = []
    dropped = 0

    for row in rows:
        prompt = make_prompt(row)
        response = row["output"].strip()

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        response_ids = tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]

        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        total_len = len(prompt_ids) + len(response_ids)

        if total_len > max_length:
            dropped += 1
            continue

        input_ids = prompt_ids + response_ids

        labels = (
            [-100] * len(prompt_ids)
            + response_ids
        )

        valid.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "source": source,
                "length": total_len,
            }
        )

    return valid, len(rows), dropped


def stratified_split(samples, val_ratio, seed):
    by_source = {}

    for x in samples:
        by_source.setdefault(x["source"], []).append(x)

    train = []
    val = []

    for source, rows in sorted(by_source.items()):
        rng = random.Random(seed)
        rng.shuffle(rows)

        val_n = round(len(rows) * val_ratio)

        val.extend(rows[:val_n])
        train.extend(rows[val_n:])

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


class SFTDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        labels = []
        attention_mask = []

        for x in batch:
            n = len(x["input_ids"])
            pad_n = max_len - n

            input_ids.append(
                x["input_ids"]
                + [self.pad_token_id] * pad_n
            )

            labels.append(
                x["labels"]
                + [-100] * pad_n
            )

            attention_mask.append(
                [1] * n
                + [0] * pad_n
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    losses = []

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            out = model(**batch)

        losses.append(out.loss.detach().float().item())

    model.train()

    if not losses:
        return float("nan")

    return sum(losses) / len(losses)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    print("=== configuration ===")
    print(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    py_rows, py_total, py_drop = load_one_file(
        args.py_data,
        "python",
        tokenizer,
        args.max_length,
    )

    cpp_rows, cpp_total, cpp_drop = load_one_file(
        args.cpp_data,
        "cpp",
        tokenizer,
        args.max_length,
    )

    samples = py_rows + cpp_rows

    train_rows, val_rows = stratified_split(
        samples,
        args.val_ratio,
        args.seed,
    )

    print("\n=== dataset ===")
    print(
        f"python: total={py_total}, "
        f"keep={len(py_rows)}, drop={py_drop}"
    )
    print(
        f"cpp:    total={cpp_total}, "
        f"keep={len(cpp_rows)}, drop={cpp_drop}"
    )
    print("combined valid =", len(samples))
    print("train =", len(train_rows))
    print("val   =", len(val_rows))

    metadata = {
        "args": vars(args),
        "python_total": py_total,
        "python_valid": len(py_rows),
        "python_dropped": py_drop,
        "cpp_total": cpp_total,
        "cpp_valid": len(cpp_rows),
        "cpp_dropped": cpp_drop,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }

    with open(
        Path(args.output_dir) / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    collator = DataCollator(
        tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        SFTDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    val_loader = DataLoader(
        SFTDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print("\n=== loading model ===")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    device = torch.device("cuda")
    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(
        len(train_loader)
        / args.grad_accum
    )

    planned_steps = (
        steps_per_epoch * args.epochs
    )

    if args.max_steps > 0:
        total_steps = min(
            planned_steps,
            args.max_steps,
        )
    else:
        total_steps = planned_steps

    warmup_steps = round(
        total_steps * args.warmup_ratio
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\n=== optimization ===")
    print("micro batches / epoch =", len(train_loader))
    print("optimizer steps / epoch =", steps_per_epoch)
    print("planned optimizer steps =", planned_steps)
    print("actual max steps =", total_steps)
    print("warmup steps =", warmup_steps)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0
    running_loss = 0.0

    stop_training = False

    for epoch in range(args.epochs):
        print(
            f"\n===== epoch {epoch + 1}/{args.epochs} ====="
        )

        for batch_idx, batch in enumerate(train_loader):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                out = model(**batch)

                raw_loss = out.loss

                loss = raw_loss / args.grad_accum

            loss.backward()

            running_loss += raw_loss.detach().float().item()
            micro_step += 1

            should_step = (
                micro_step % args.grad_accum == 0
                or batch_idx == len(train_loader) - 1
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in model.parameters()
                        if p.requires_grad
                    ],
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                if (
                    global_step % args.log_steps == 0
                    or global_step == 1
                ):
                    avg_loss = (
                        running_loss
                        / max(micro_step, 1)
                    )

                    mem = (
                        torch.cuda.max_memory_allocated()
                        / 1024**3
                    )

                    print(
                        f"step={global_step} "
                        f"loss={avg_loss:.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"max_gpu_mem={mem:.2f} GiB"
                    )

                    running_loss = 0.0
                    micro_step = 0

                if (
                    args.max_steps > 0
                    and global_step >= args.max_steps
                ):
                    stop_training = True
                    break
        micro_step = 0
        running_loss = 0.0

        if stop_training:
            break

        print("\nRunning validation...")
        val_loss = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"epoch={epoch + 1} "
            f"val_loss={val_loss:.6f}"
        )

        epoch_dir = (
            Path(args.output_dir)
            / f"checkpoint-epoch-{epoch + 1}"
        )

        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        print("saved:", epoch_dir)

    final_dir = (
        Path(args.output_dir)
        / "final_adapter"
    )

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n=== finished ===")
    print("optimizer steps =", global_step)
    print("saved =", final_dir)


if __name__ == "__main__":
    main()
