# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Finetuning on Huggingface wo ZeRO and deepspeed
# MAGIC
# MAGIC This code was tested on MLR 13.3
# MAGIC Based on: https://github.com/microsoft/DeepSpeedExamples/blob/902a0f6b2b4a87c8048c0ab3a823b749c0e218ab/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py

# COMMAND ----------

# MAGIC # peft==0.4.0  bitsandbytes==0.39.1
# MAGIC %pip install deepspeed==0.9.4 

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

# load libs
import os
from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
from transformers import (
  AutoModelForCausalLM, AutoTokenizer, 
  DataCollatorForLanguageModeling
) 
import torch
from datasets import load_dataset
import mlflow
from torch.utils.data import RandomSampler, DataLoader

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

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
# MAGIC ## Dataset
# MAGIC
# MAGIC We will use the [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Args

# COMMAND ----------

#global_batch_size = 1
micro_batch_size = 1
offload_cpu = True
learning_rate = 0.01
num_epochs = 2

tensorboard_path = '/local_disk0/tensorboard'
tf_path = f'/home/{username}/deepspeed/tf_logs'
dbutils.fs.mkdirs(tf_path)
dbfs_tf_path = f'/dbfs{tf_path}'

dbfs_tf_path = tensorboard_path

datasets_cache = '/home/{username}/datasets_cache'
dbutils.fs.mkdirs(tf_path)
dbfs_datasets_cache = f'/dbfs{datasets_cache}'


# COMMAND ----------

# MAGIC %md # Deepspeed Configuration 

# COMMAND ----------

# Deepspeed config

device = "cpu" if offload_cpu else "none"

# When doing distributed train we can leave out train size
# So that deepspeed will work it out otherwise can get config issues
# "train_batch_size": global_batch_size,
deepspeed_dict = {
    "bf16": {
        "enabled": "auto"
    },

    "optimizer": {
        "type": "Adamw",
        "params": {
            "lr": learning_rate,
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-05,
            "warmup_num_steps": 100
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": device,
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "tensorboard": {
      "enabled": True,
      "output_path": dbfs_tf_path,
      "job_name": "finetune_llama_2_7b"
    },

    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 1,
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "wall_clock_breakdown": True
}


# COMMAND ----------

# MAGIC %md ## Data Setup
# COMMAND ----------

def setup_data(tokenizer, dataset):
    """
    Args:
      tokenizer: hf tokenizer
      dataset: hf dataset
    """
    

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

    dataset = dataset.map(tokenize_function, batched=True)

    dataset = dataset.remove_columns(['instruction', 'response', 'context', 'text', 'category'])

    return dataset

# COMMAND ----------

# MAGIC %md # Main Train Function

# COMMAND ----------

from deepspeed import get_accelerator

def train(*, dataset):
    
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    
    # need to set these so that distributed workers can access the data and the key
    os.environ['HF_DATASETS_CACHE'] = datasets_cache
    os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

    
    mlflow.set_experiment(experiment_path)

    model_id = "meta-llama/Llama-2-7b-hf"
    revision = "351b2c357c69b4779bde72c0e7f7da639443d904"

    #model_id = 'databricks/dolly-v2-3b'
    #revision = 'f6c9be08f16fe4d3a719bee0a4a7c7415b5c65df'

    ## setup the first part

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    device = torch.device(get_accelerator().device_name())

    #device = torch.device("cuda")

    global_rank = torch.distributed.get_rank()

    #torch.distributed.barrier()

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #quantization_config=bnb_config,
        device_map = {"":int(local_rank)},
        revision=revision,
        torch_dtype = torch.bfloat16,
        #quantization_config=bnb_config,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # We don't want the trainer to auto distribute the model
    # I think?

    train_dataset = setup_data(tokenizer, dataset)
    train_sampler = RandomSampler(train_dataset)


    # setup trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
       train_dataset,
       collate_fn = data_collator,
       sampler = train_sampler,
       batch_size = micro_batch_size
    )

    # optimizer
    AdamOptimizer = DeepSpeedCPUAdam if offload_cpu else FusedAdam
    optimizer = AdamOptimizer(model.parameters(),
                              lr=learning_rate,
                              betas=(0.9, 0.95))


    # model, optimizer
    initialised_var  = deepspeed.initialize(
       model = model,
       optimizer = optimizer,
       dist_init_required=False,
       config = deepspeed_dict
    )

    model = initialised_var[0]
    optimizer = initialised_var[1]

    for epoch in range(num_epochs):
      model.train()

      for step, batch in enumerate(train_dataloader):
          batch.to(device)
          outputs = model(**batch, use_cache=False)

          loss = outputs.loss

          model.backward(loss)
          model.step()

    return model

# COMMAND ----------

# MAGIC %md
# MAGIC # Run training

# COMMAND ----------

dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train", cache_dir = dbfs_datasets_cache)

# COMMAND ----------

distributor = DeepspeedTorchDistributor(numGpus=8, nnodes=1, localMode=True, 
                                        useGpu=True, deepspeedConfig = deepspeed_dict)

completed_trainer = distributor.run(train, dataset=dataset)
