"""
As i mantioned before i am going to use LoRA for 
fine-tune of Qwen-7B model to use it as agent. In 
that file i will mention configuration and use it \
as studying script on kaggle
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,  # type: ignore
    get_peft_model,  # type: ignore
    prepare_model_for_kbit_training,  # type: ignore
)

# configuration for pc

# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# LORA_OUTDIR = "adapters/qwen_lora"
# TRAIN_FILE = "data/processed/train_pairs.jsonl"
# VAL_FILE = "data/processed/val_pairs.jsonl"
# MAX_SEQ_LENGTH = 1024
# BATCH_SIZE = 2
# GRAD_ACC_STEPS = 4
# EPOCHS_OR_STEPS = {"max_steps": 3000}
# LR_LORA = 2e-4
# LR_BASE = 0

# configuration for kaggle

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_OUTDIR = "/kaggle/working/adapters/qwen_lor"
TRAIN_FILE = "/kaggle/input/datasets/belyaevgleb/avito-model-dataset/train_pairs.jsonl"
VAL_FILE = "/kaggle/input/datasets/belyaevgleb/avito-model-dataset/val_pairs.jsonl"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACC_STEPS = 4
EPOCHS_OR_STEPS = {"max_steps": 3000}
LR_LORA = 2e-4
LR_BASE = 0


@dataclass
class LoraArguments:
    r: int = 16
    lora_alpha: int = 32
    target_modules = ["q_proj", "v_proj", "gate_proj"]
    lora_dropout = 0.05
    bias = "none"
    task_type = "CASUAL_LM"


def load_jsonl_to_dataset(path: str) -> Dataset:
    """Reads a JSONL file and converts it into a Hugging Face Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"File {path} is empty or not found.")
    return Dataset.from_dict({k: [rec[k] for rec in records] for k in records[0]})


def build_chat_prompt(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Encodes the conversation into ChatML format.
    Masks the prompt tokens with -100 so cross-entropy loss is computed
    strictly on the assistant's response.
    """
    system_text = ""
    user_text = ""
    assistant_text = ""
    for msg in example["messages"]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content
    prompt_str = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    response_str = f"{assistant_text}<|im_end|>"
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    response_ids = tokenizer.encode(response_str, add_special_tokens=False)
    input_ids = prompt_ids + response_ids
    # Prompt is masked with -100; completion keeps its token IDs
    labels = [-100] * len(prompt_ids) + response_ids
    # Truncate to MAX_SEQ_LENGTH if necessary
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    print(
        f"Tokenizer loaded: pad_token={tokenizer.pad_token} (id={tokenizer.pad_token_id})"
    )
    print("Loading datasets...")
    train_ds = load_jsonl_to_dataset(TRAIN_FILE)
    val_ds = load_jsonl_to_dataset(VAL_FILE)
    train_ds = train_ds.map(
        lambda ex: build_chat_prompt(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda ex: build_chat_prompt(ex, tokenizer),
        remove_columns=val_ds.column_names,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading 8-bit base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
    )
    print("Базовая модель успешно загружена!")
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    lora_params = [
        p for n, p in model.named_parameters() if "lora" in n and p.requires_grad
    ]
    base_params = [
        p for n, p in model.named_parameters() if "lora" not in n and p.requires_grad
    ]
    optimizer_grouped_parameters = [
        {"params": lora_params, "lr": LR_LORA},
    ]
    if base_params:
        optimizer_grouped_parameters.append({"params": base_params, "lr": LR_BASE})
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999))
    training_args = TrainingArguments(
        output_dir="/kaggle/working/tmp_checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        max_steps=1200,
        learning_rate=LR_LORA,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        optimizers=(optimizer, None),
    )
    print("Starting training...")
    trainer.train()
    print(f"Saving LoRA adapter to {LORA_OUTDIR}...")
    Path(LORA_OUTDIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LORA_OUTDIR)
    tokenizer.save_pretrained(LORA_OUTDIR)
    print("✅ Training successfully completed.")


if __name__ == "__main__":
    main()
