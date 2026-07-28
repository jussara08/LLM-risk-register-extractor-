"""
QLoRA fine-tuning of a small open-source LLM on the risk narrative -> JSON task.

Run this on a machine/notebook with GPU access (e.g. Colab, a cloud GPU box).
It will NOT run in a CPU-only environment in reasonable time.

Usage:
    python scripts/train.py \
        --data data/train.jsonl \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --output_dir models/risk-register-lora \
        --epochs 3

Design notes:
- Uses 4-bit quantization (QLoRA) so this is trainable on a single consumer
  GPU (e.g. a 16-24GB card) or a free/cheap Colab GPU instance.
- Each training example is formatted as an instruction/response pair, with
  the assistant response being the raw JSON string (matching what we want
  at inference time -- no markdown fences, no preamble).
- LoRA is applied to attention projection layers only, which is standard
  practice and keeps the trainable parameter count small (~1% of base model).
"""

import argparse
import json

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from schema_config import SYSTEM_PROMPT


def format_example(example: dict, tokenizer) -> dict:
    """Turns a {narrative, output} row into a chat-formatted training string."""
    narrative = example["narrative"]
    output_json = json.dumps(example["output"], ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract a structured risk register entry from this narrative:\n\n{narrative}"},
        {"role": "assistant", "content": output_json},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output_dir", default="models/risk-register-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(lambda ex: format_example(ex, tokenizer))
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        dataset_text_field="text",
        max_seq_length=1024,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
