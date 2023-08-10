# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Testing out SFT step
# MAGIC
# MAGIC We need updates to transformers and also accelerate to get this work properly\
# MAGIC As they are things that are installed on the os level in the prebuilt AMI it is best to update these on the os layer via init script 
# MAGIC
# MAGIC This code was tested on MLR 14.0

# COMMAND ----------

# MAGIC %pip install transformers -U accelerate -U peft trl bitsandbytes

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

# TODO databricks secrets link and hf token guide

import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Extra Notes**
# MAGIC
# MAGIC We need to pack samples into a ConstantLengthDataset to optimise the training process\
# MAGIC See: https://huggingface.co/docs/trl/main/en/sft_trainer#packing-dataset-constantlengthdataset

# COMMAND ----------

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Functions

# COMMAND ----------

# DBTITLE 1,Functions
def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, dataset_name, 
                    subset, split, num_workers, 
                    streaming, size_valid_set, local_disk_config,
                    shuffle_buffer, seq_length):
    """
    Create the dataset for the trainer
    args required:
      dataset_name
      subset
      split
      num_workers / streaming
      size_valid_set
      shuffle_buffer
      seq_length
    """
    dataset = load_dataset(
        dataset_name,
        data_dir=subset,
        split=split,
        use_auth_token=True,
        download_config=local_disk_config,
        num_proc=num_workers if not streaming else None,
        streaming=streaming,
    )
    if streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(size_valid_set)
        train_data = dataset.skip(size_valid_set)
        train_data = train_data.shuffle(buffer_size=shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Parameters

# COMMAND ----------

# DBTITLE 1,Params
model_name = 'meta-llama/Llama-2-7b-hf'

# See 
lora_r=8
lora_dropout=0.05
lora_alpha=16

# output_dir
output_dir = '/tmp/lora_test'
dbutils.fs.mkdirs(output_dir)
dbfs_output_dir = f'/dbfs{output_dir}'

per_device_train_batch_size = 4
gradient_accumulation_steps = 2
per_device_eval_batch_size = 1
learning_rate = 1e-4
logging_steps = 10
max_steps = 500
log_with = 'wandb' # tbd - to switch to mlflow
save_steps = 10
group_by_length = False
lr_scheduler_type = 'cosine'
num_warmup_steps = 100
optimizer_type = 'paged_adamw_32bit'

# dataset params
dataset_name = "lvwerra/stack-exchange-paired"
subset = "data/finetune"
split = 'train'
num_workers = 4
streaming = True
size_valid_set = 4000
shuffle_buffer = 5000
seq_length = 1024

# COMMAND ----------

# DBTITLE 1,extra download config for huggingface
from datasets import DownloadConfig

local_disk_config = DownloadConfig(cache_dir='/local_disk0/datasets_cache')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Setting Up Training Loop

# COMMAND ----------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False


# COMMAND ----------

# We need to explore more this config with the target_modules
# See Tim Dettmers discussion

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# COMMAND ----------

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    max_steps=max_steps,
    #report_to=log_with,
    save_steps=save_steps,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=num_warmup_steps,
    optim=optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft_llama2",
)


# COMMAND ----------

train_dataset, eval_dataset = create_datasets(tokenizer, dataset_name, 
                    subset, split, num_workers, 
                    streaming, size_valid_set, local_disk_config,
                    shuffle_buffer, seq_length)

# COMMAND ----------

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=True,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)


# COMMAND ----------

trainer.train()

# COMMAND ----------

trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
