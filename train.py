"""
QLoRA fine‑tune for Llama‑3 on a big DS‑QA corpus.
Run on a single RTX 3060 with:
accelerate launch train.py --batch_size 1 --gradient_accumulation_steps 32
"""
import argparse, os, json, torch, datasets, transformers, peft, bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--dataset", default="data/big_qa.jsonl")
    p.add_argument("--val", default="data/ds_qa_val.jsonl")
    p.add_argument("--output_dir", default="outputs/llama3-ds-qlora")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()

def load_jsonl(path):
    ds = datasets.load_dataset("json", data_files={"data": path}, split="data")
    return ds

def formatting(example):
    return f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}\n"

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_jsonl(args.dataset)
    val_ds   = load_jsonl(args.val)

    def tokenize(batch):
        text = formatting(batch)
        out = tokenizer(
            text, truncation=True, max_length=args.max_seq_length,
            padding="max_length"  # pad for efficient packing
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(tokenize,   remove_columns=val_ds.column_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    fourbit = dict(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", **fourbit
    )

    if args.flash_attn:
        transformers.utils.import_utils._add_flash_attn_monkey_patch()

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, lora_cfg)

    optim = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)
    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optim.step(); optim.zero_grad()
            if accelerator.is_local_main_process and global_step % 50 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
            global_step += 1
        # simple validation loop
        model.eval(); val_loss = 0; n = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch); val_loss += outputs.loss.item(); n += 1
        if accelerator.is_local_main_process:
            print(f"epoch {epoch} val_loss {val_loss/n:.4f}")
        model.train()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
