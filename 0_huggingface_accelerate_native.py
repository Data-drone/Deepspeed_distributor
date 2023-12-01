# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Finetuning on Huggingface wo ZeRO and deepspeed
# MAGIC
# MAGIC This code was tested on MLR 14.2
# MAGIC Lets see how we can scale the native library functions \
# MAGIC Note that this will just work on single node \
# MAGIC But this is how we can leverage Accelerate driven HF Trainer on Databricks

# COMMAND ----------

# MAGIC %pip install peft==0.6.0 deepspeed==0.12.1 bitsandbytes==0.41.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# we can create a generic setup
import accelerate
from datasets import load_dataset
import os


# COMMAND ----------

# Databricks configuration and MLflow setup
browser_host = spark.conf.get("spark.databricks.workspaceUrl")
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/deepspeed-distributor'

datasets_cache = f'/home/{username}/datasets_cache'
model_cache_root = f'/home/{username}/hf_models'
dbfs_datasets_cache = f'/dbfs{datasets_cache}'


# COMMAND ----------

# we need to make sure that a whole bunch of parameters are aligned between deepspeed and hf

shared_parameters = {
   "gradient_accumulation_steps": 1,
   "gradient_clipping": 0.3,
   "per_device_batch_size": 4,
   "learning_rate": 2e-4,
   "warmup_steps": 10
}

# COMMAND ----------

def setup_params(shared_parameters:dict, mlflow_run_name: str='single_run', deepspeed_config=None):

    # setup training arguments
    # We do this inside a function so that we don't initialise cuda before Accelerate takes over 

    from peft import LoraConfig
    from transformers import TrainingArguments

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
    per_device_train_batch_size = shared_parameters['per_device_batch_size']
    gradient_accumulation_steps = shared_parameters['gradient_accumulation_steps']
    optim = "paged_adamw_32bit"
    save_steps = 50
    logging_steps = 10
    learning_rate = shared_parameters['learning_rate']
    max_grad_norm = shared_parameters['gradient_clipping']
    max_steps = 100
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
        run_name=mlflow_run_name,
        deepspeed=deepspeed_config
    )

    return peft_config, training_arguments


# COMMAND ----------

# DBTITLE 1,Load Dataset
dataset_name = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_name, split="train", cache_dir = dbfs_datasets_cache)

# COMMAND ----------

def setup_data(tokenizer, dataset):

    # This is our dataset setup function
    
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

def train(peft_config, training_arguments, dataset, distributor=True, deepspeed=False):

    # This is our main training function

    from transformers import (
       BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer,
       DataCollatorForLanguageModeling, Trainer
    )
    from peft import get_peft_model

    import mlflow
    import torch

    os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = f'/Users/{username}/dist-torch'
    #os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = 'True'
    
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    if distributor:
      os.environ['NCCL_IB_DISABLE'] = '1'
      os.environ['NCCL_P2P_DISABLE'] = '1'

    mlflow.set_registry_uri('databricks')
    mlflow.set_experiment(experiment_path)

    model_id = "meta-llama/Llama-2-7b-hf"
    revision = "351b2c357c69b4779bde72c0e7f7da639443d904"
    model_path = f'/dbfs{model_cache_root}/llama_2_7b'

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    device_map_var = None if deepspeed else {"":int(os.environ.get("LOCAL_RANK"))}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device_map_var,
        revision=revision,
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        cache_dir=model_path,
        local_files_only=True
    )

    # Do we still need to set these flags?
    model.is_parallelizable = False
    model.model_parallel = False

    model = get_peft_model(model, peft_config)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
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

    # if we return the trainer object we get an error
    return 'done'

# COMMAND ----------

# DBTITLE 1,Launch with Accelerate (single node only)
from accelerate import notebook_launcher

def accelerate_train():
   
    peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                   mlflow_run_name='accelerate_run')
    trainer = train(peft_config, training_arguments, dataset, distributor=True)

    return trainer

accelerate.utils.write_basic_config()

num_gpus_on_driver = 1
notebook_launcher(accelerate_train, num_processes=num_gpus_on_driver)

# COMMAND ----------

# DBTITLE 1,Launch with TorchDistributor
from pyspark.ml.torch.distributor import TorchDistributor

def accelerate_train():
   
    peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                   mlflow_run_name='torch_distributor')
    trainer = train(peft_config, training_arguments, dataset, distributor=True)

    return trainer

# Test this code with TorchDistributor?
num_gpus_per_node = 1
num_nodes = 2
num_processes = num_gpus_per_node * num_nodes

distributor = TorchDistributor(num_processes=num_processes, 
                               local_mode=True, use_gpu=True)
completed_trainer = distributor.run(accelerate_train)

# COMMAND ----------

# DBTITLE 1,Launch with Deepspeed Distributor
deepspeed_dict = {
    "bf16": {
        "enabled": "auto"
    },

    "optimizer": {
        "type": "Adamw",
        "params": {
            "lr": shared_parameters['learning_rate'],
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": shared_parameters['learning_rate'],
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": 'cpu'
        },
        "offload_param": {
            "device": 'cpu'
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
      "output_path": '/local_disk0/tensorboard',
      "job_name": "finetune_llama_2_7b"
    },

    "gradient_accumulation_steps": shared_parameters['gradient_accumulation_steps'],
    "gradient_clipping": shared_parameters['gradient_clipping'],
    "steps_per_print": 1,
    "train_micro_batch_size_per_gpu": shared_parameters['per_device_batch_size'],
    "train_batch_size": "auto",
    "wall_clock_breakdown": True
}

# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

def deepspeed_train():
   
    peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                   mlflow_run_name='deepspeed_distributor_w_config',
                                                   deepspeed_config=deepspeed_dict)
    trainer = train(peft_config, training_arguments, dataset, 
                    distributor=True, deepspeed=True)

    return trainer


distributor = DeepspeedTorchDistributor(numGpus=1, nnodes=2, localMode=False, 
                                        useGpu=True, deepspeedConfig = deepspeed_dict)

completed_trainer = distributor.run(deepspeed_train)

# COMMAND ----------