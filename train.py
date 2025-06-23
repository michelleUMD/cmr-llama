import glob
import os
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
import torch
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
import xml.etree.ElementTree as ET

import xmltodict
import json

from trl import SFTTrainer
import pandas as pd

# Import from utils.py
from utils import (
    ScriptArguments,
    template,
    remove_tags,
    prep_xml,
    parse_xml,
    gen_batches_cmr,
    create_and_prepare_model
)

"""
Example usage for Llama 3.3:
python train.py \
    --model_name "meta-llama/Meta-Llama-3.3-8B" \
    --model_type "llama3.3" \
    --use_4bit True \
    --gradient_checkpointing True \
    --output_dir "./results_llama33" \
    --bf16 False \
    --fp16 True

Example usage for Llama 3.1:
python train.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --model_type "llama3.1" \
    --use_4bit True \
    --output_dir "./results_llama31" \
    --bf16 True

Example usage for QwQ-32B:
python train.py \
    --model_name "Qwen/QwQ-32B" \
    --model_type "qwq32b" \
    --use_4bit True \
    --gradient_checkpointing True \
    --output_dir "./results_qwq32b" \
    --bf16 False \
    --fp16 True
"""

torch.manual_seed(42)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_steps=script_args.warmup_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    report_to=script_args.report_to,
    gradient_checkpointing=script_args.gradient_checkpointing,
    # evaluation_strategy="steps",
    # eval_steps=1,
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)

# Print model configuration for debugging
print(f"Model name: {script_args.model_name}")
print(f"Model type: {script_args.model_type}")
print(f"Using 4-bit quantization: {script_args.use_4bit}")
print(f"Gradient checkpointing: {script_args.gradient_checkpointing}")

train_dataset = Dataset.from_generator(lambda: gen_batches_cmr("/data/aiiih/projects/fangm/nlp_data/Train/*/*.xml", phase='train'))
tokenizer.padding_side = 'left'

# Apply gradient checkpointing if enabled
if script_args.gradient_checkpointing:
    # For BF16 with gradient checkpointing, we need to be careful with certain operations
    if script_args.bf16 and script_args.model_type.lower() == "llama3.3":
        print("Using gradient checkpointing with BF16 for Llama 3.3 - setting attn_implementation to sdpa")
        # This is already handled in model creation with attn_implementation="sdpa"
    elif script_args.model_type.lower() == "qwq32b":
        print("Using gradient checkpointing with QwQ-32B")
        # QwQ-32B specific settings if needed
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

# Add a warning about potential BF16 issues
if script_args.bf16 and script_args.gradient_checkpointing:
    print("=" * 80)
    print("WARNING: Using BF16 with gradient checkpointing may cause 'triu_tril_cuda_template' errors")
    print("If you encounter this error, try using --fp16 True --bf16 False instead")
    print("=" * 80)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=validation_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.neftune_noise_alpha = None
trainer.train()

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    # Load and merge the model
    print(f"Loading model from {output_dir} for merging...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, 
        device_map="auto", 
        torch_dtype=torch.bfloat16 if script_args.bf16 else torch.float16
    )
    
    print("Merging model weights...")
    model = model.merge_and_unload()

    # Save the merged model
    output_merged_dir = os.path.join(script_args.output_dir, f"final_merged_{script_args.model_type}_checkpoint")
    print(f"Saving merged model to {output_merged_dir}...")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    
    # Save tokenizer with the model
    tokenizer.save_pretrained(output_merged_dir)
    print(f"Model and tokenizer saved to {output_merged_dir}")
    
    # Print information about the saved model
    print(f"Model type: {script_args.model_type}")
    print(f"Original model: {script_args.model_name}")
    print(f"Training completed and model merged successfully!")
