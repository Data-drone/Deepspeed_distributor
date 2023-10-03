# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune llama-2-7b-hf with QLORA
# MAGIC
# MAGIC [Llama 2](https://huggingface.co/meta-llama) is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is trained with 2T tokens and supports context length window upto 4K tokens. [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) is the 7B pretrained model, converted for the Hugging Face Transformers format.
# MAGIC
# MAGIC This is to fine-tune [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) models on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 14.0 GPU ML Runtime
# MAGIC - Instance: `g5.8xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient finetuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). We will also install `einops` as it is a requirement to load Falcon models.

# COMMAND ----------

# MAGIC %pip install peft==0.4.0
# MAGIC %pip install bitsandbytes==0.40.1 trl==0.4.7

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import huggingface_hub

# Login to Huggingface to get access to the model
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks config

# COMMAND ----------

import mlflow

browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/deepspeed-distributor'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC
# MAGIC We will use the [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

# COMMAND ----------

from datasets import load_dataset

dataset_name = "databricks/databricks-dolly-15k"
#dataset = load_dataset(dataset_name, split="train")

# COMMAND ----------

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

# COMMAND ----------

from torch.utils.data import DataLoader
from accelerate import Accelerator

def get_dataloaders(dataset, accelerator, tokenizer, batch_size = 4):
   
    #dataset = load_dataset(dataset_name, split="train")
    
    def tokenizer_function(example):
       outputs = tokenizer(example["text"], truncation=True, max_length=128,
                           return_overflowing_tokens=True)
       
       sample_map = outputs.pop("overflow_to_sample_mapping")
       for key, values in example.items():
          outputs[key] = [values[i] for i in sample_map]
       return outputs
    
    with accelerator.main_process_first():
       dataset = dataset.map(apply_prompt_template)
       tokenized_datasets = dataset.map(
          tokenizer_function,
          batched=True,
          writer_batch_size=100,
          remove_columns=["instruction", "context", "response", "category"]
       )
    
    def collate_fn(examples):
       max_length = 1000
       return tokenizer.pad(examples,
                            max_length=max_length,
                            return_tensors="pt"
                            )
    
    train_dataloader = DataLoader(
       tokenized_datasets, shuffle=True, collate_fn=collate_fn, 
       batch_size=batch_size, drop_last=True
    )

    return train_dataloader

# COMMAND ----------

def get_mapped_dataset(dataset):
   
   dataset = dataset.map(apply_prompt_template)

   return dataset

# COMMAND ----------

# basic dataloader - lets move the tokenization to the sft loader?

def get_untokenized_dataloaders(dataset, accelerator, tokenizer, batch_size = 4):

    with accelerator.main_process_first():
       dataset = dataset.map(apply_prompt_template)

    train_dataloader = DataLoader(
       dataset, shuffle=True,
       batch_size=batch_size, drop_last=True
    )

    return train_dataloader

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the model
# MAGIC
# MAGIC In this section we will load the [LLaMAV2](), quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-2-7b-hf"
revision = "351b2c357c69b4779bde72c0e7f7da639443d904"

local_stash = '/local_disk0/llama-2-7b'

snapshot_download(model_id, local_dir=local_stash)

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

def create_model():
  
  tokenizer = AutoTokenizer.from_pretrained(local_stash, local_files_only=True)
  tokenizer.pad_token = tokenizer.eos_token

  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
  )

  model = AutoModelForCausalLM.from_pretrained(
    local_stash,
    quantization_config=bnb_config,
    revision=revision,
    device_map={'':torch.cuda.current_device()},
    local_files_only=True,
  )
  model.config.use_cache = False

  return tokenizer, model

# COMMAND ----------

# MAGIC %md
# MAGIC Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. Therefore we will add `dense`, `dense_h_to_4_h` and `dense_4h_to_h` layers in the target modules in addition to the mixed query key value layer.

# COMMAND ----------

from peft import LoraConfig

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the trainer
# MAGIC Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# COMMAND ----------

from transformers import TrainingArguments

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
    group_by_length=False,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model

# COMMAND ----------
# fix for dataset issue?
dataset = load_dataset(dataset_name, split="train")

# COMMAND ----------

from trl import SFTTrainer
from trl.trainer.utils import ConstantLengthDataset
from pyspark.ml.torch.distributor import TorchDistributor
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
import os

def main():
   
   accelerator = Accelerator(device_placement=False)
   
   os.environ['DATABRICKS_HOST'] = db_host
   os.environ['DATABRICKS_TOKEN'] = db_token

   mlflow.set_experiment(experiment_path)

   max_seq_length = 512

   tokenizer, model = create_model()

   train_dataset = get_mapped_dataset(dataset)
  
   # setup data loader
   #pt_loader = DataLoader(dataset)

   train_loader = ConstantLengthDataset(tokenizer, train_dataset, 
                                        dataset_text_field='text')

   #train_loader = prepare_data_loader(pt_loader, device=torch.cuda.current_device())
   train_loader = accelerator.prepare(train_loader)

   trainer = SFTTrainer(
    model=model,
    train_dataset=train_loader,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
  )
   
   for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

   trainer.train()

   return trainer

# COMMAND ----------

distributor = TorchDistributor(num_processes=2, local_mode=True, use_gpu=True)
completed_trainer = distributor.run(main)

# COMMAND ----------

os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# COMMAND ----------

# MAGIC %sh torchrun --nnodes 1 --nproc-per-node 2 /Workspace/Users/brian.law@databricks.com/.ide/Deepspeed_distributor-69ddcada/train_script.py

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the LORA model

# COMMAND ----------

trainer.save_model("/local_disk0/llamav2-7b-lora-fine-tune")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the fine tuned model to MLFlow

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig

peft_model_id = "/local_disk0/llamav2-7b-lora-fine-tune"
config = PeftConfig.from_pretrained(peft_model_id)

from huggingface_hub import snapshot_download
# Download the Llama-2-7b-hf model snapshot from huggingface
snapshot_location = snapshot_download(repo_id=config.base_model_name_or_path)


# COMMAND ----------

import mlflow
class LLAMAQLORA(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'])
    self.tokenizer.pad_token = tokenizer.eos_token
    config = PeftConfig.from_pretrained(context.artifacts['lora'])
    base_model = AutoModelForCausalLM.from_pretrained(
      context.artifacts['repository'], 
      return_dict=True, 
      load_in_4bit=True, 
      device_map={"":0},
      trust_remote_code=True,
    )
    self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])
  
  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [1.0])[0]
    max_tokens = model_input.get("max_tokens", [100])[0]
    batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = self.model.generate(
          input_ids = batch.input_ids, 
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import mlflow

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=LLAMAQLORA(),
        artifacts={'repository' : snapshot_location, "lora": peft_model_id},
        pip_requirements=["torch", "transformers", "accelerate", "einops", "loralib", "bitsandbytes", "peft"],
        input_example=pd.DataFrame({"prompt":["what is ML?"], "temperature": [0.5],"max_tokens": [100]}),
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

import mlflow
import pandas as pd


prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

### Response: """
# Load model as a PyFuncModel.
run_id = run.info.run_id
logged_model = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(logged_model)

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Predict on a Pandas DataFrame.