# scripts/vyvo_finetune.py
"""
Minimal LoRA fine-tune of VyvoTTS (Qwen3-0.6B) on SNAC codes.
Assumes data/train_snac.jsonl from precompute_snac.py.

Usage:
  uv run scripts/vyvo_finetune.py

Outputs:
  vyvo_qwen3_lora/
"""
import ast
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Vyvo/VyvoTTS-v0-Qwen3-0.6B"

# Token constants (match inference script)
TOKENISER_LEN = 151669
SOT, EOT = 151643, 151645
SOH, EOH = TOKENISER_LEN + 3, TOKENISER_LEN + 4
PAD = TOKENISER_LEN + 7


@dataclass
class Example:
    text: str
    codes: List[int]


def collate_fn(batch, tokenizer):
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        txt = ex["text"]
        codes = ex["codes"]
        ids = tokenizer(txt, return_tensors="pt").input_ids[0]
        prompt = torch.cat([torch.tensor([SOH]), ids, torch.tensor([EOT, EOH])], dim=0)
        # Simple left-pad prompt, labels are the audio codes
        pad = torch.full((prompt.shape[0],), PAD, dtype=torch.long)
        inp = torch.cat([pad, prompt], dim=0)
        attn = torch.cat([torch.zeros_like(pad), torch.ones_like(prompt)], dim=0)
        input_ids.append(inp)
        attention_mask.append(attn)
        labels.append(torch.tensor(codes, dtype=torch.long))
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=PAD
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        ),
    }


def main():
    # Load base model in 4-bit to fit 8 GB
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=4096,
        dtype=torch.float16 if DEVICE == "cuda" else None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)
    # Note: Don't call .to(DEVICE) on quantized models

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_cfg)

    ds = load_dataset("json", data_files={"train": "../data/train_snac.jsonl"})["train"]
    dl = DataLoader(
        ds, batch_size=1, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()

    max_steps = 4000
    for step, batch in enumerate(dl, 1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % 25 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        if step >= max_steps:
            break

    out_dir = "vyvo_qwen3_lora"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✅ Saved LoRA adapters to {out_dir}")


if __name__ == "__main__":
    main()
