# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Finetuning on Huggingface
# MAGIC
# MAGIC This code was tested on MLR 13.2

# COMMAND ----------

# MAGIC %pip install peft==0.5.0
# MAGIC %pip install datasets==2.12.0 bitsandbytes==0.41.0 einops==0.6.1 trl==0.4.7
# MAGIC %pip install torch==2.0.1 accelerate==0.21.0 transformers==4.31.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Setup

# COMMAND ----------

# TODO databricks secrets link and hf token guide

import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup the experiment

# COMMAND ----------

# load libs
import os
from pyspark.ml.torch.distributor import TorchDistributor
from transformers import (
  AutoModelForCausalLM, TrainingArguments, AutoTokenizer, 
  DataCollatorForLanguageModeling, Trainer, BitsAndBytesConfig 
) 
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import mlflow
# COMMAND ----------

# MLflow settings
# We need to store out these settings to feed into our train function
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/deepspeed-distributor'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Args / Lora Configs
# COMMAND ----------
# Configuration
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'] # Choose all linear layers from the model
)

output_dir = "/local_disk0/results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 100
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
)


# COMMAND ----------

def setup_data(tokenizer, dataset):

    #dataset_name = "databricks/databricks-dolly-15k"
    #dataset = load_dataset(dataset_name, split="train")

    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    PROMPT_NO_INPUT_FORMAT = """{intro}
    
    {instruction_key}
    {instruction}
    
    {response_key}
    {response}
    
    {end_key}""".format(
      intro=INTRO_BLURB,
      instruction_key=INSTRUCTION_KEY,
      instruction="{instruction}",
      response_key=RESPONSE_KEY,
      response="{response}",
      end_key=END_KEY
    )
    
    PROMPT_WITH_INPUT_FORMAT = """{intro}
    
    {instruction_key}
    {instruction}
    
    {input_key}
    {input}
    
    {response_key}
    {response}
    
    {end_key}""".format(
      intro=INTRO_BLURB,
      instruction_key=INSTRUCTION_KEY,
      instruction="{instruction}",
      input_key=INPUT_KEY,
      input="{input}",
      response_key=RESPONSE_KEY,
      response="{response}",
      end_key=END_KEY
    )
    
    def apply_prompt_template(examples):
      instruction = examples["instruction"]
      response = examples["response"]
      context = examples.get("context")
    
      if context:
        full_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
      else:
        full_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
      return { "text": full_prompt }
    
    dataset = dataset.map(apply_prompt_template)

    # need to add in processing to make it tokenized?
    def tokenize_function(allEntries):
      return tokenizer(allEntries['text'], truncation=True, max_length=512,)

    dataset = dataset.map(tokenize_function)

    return dataset
# COMMAND ----------

def train(peft_config, training_arguments, dataset):
    
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    mlflow.set_experiment(experiment_path)

    model_id = "meta-llama/Llama-2-7b-hf"
    revision = "351b2c357c69b4779bde72c0e7f7da639443d904"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #quantization_config=bnb_config,
        device_map = {"":int(os.environ.get("LOCAL_RANK"))},
        revision=revision,
        quantization_config=bnb_config,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # We don't want the trainer to auto distribute the model
    # I think?
    model.is_parallelizable = False
    model.model_parallel = False

    model = get_peft_model(model, peft_config)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = setup_data(tokenizer, dataset)

    # setup trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=train_dataset
        #eval_dataset=val_dataset,
    )

    trainer.train()

    return trainer

# COMMAND ----------


dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train")

# COMMAND ----------

distributor = TorchDistributor(num_processes=1, local_mode=True, use_gpu=True)

completed_trainer = distributor.run(train, peft_config, training_arguments, dataset)
