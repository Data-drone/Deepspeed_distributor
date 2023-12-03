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
    #max_steps = 100
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
        num_train_epochs = 2,
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

# DBTITLE 1,Function to reformat dataset to stanford alpaca format
# MAGIC %run ./dataset_prep/setup_dataset

# COMMAND ----------

# DBTITLE 1,HuggingFace Trainer train loop
# MAGIC %run ./train_loops/hf_trainer_loop

# COMMAND ----------

# DBTITLE 1,Launch with Accelerate (single node only)
from accelerate import notebook_launcher

def accelerate_train(mlflow_run_name:str='accelerate_run', deepspeed=None):

    peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                   mlflow_run_name=mlflow_run_name)
    trainer = train(peft_config, training_arguments, dataset, distributor=True)

    return trainer

# we need to write a config
#accelerate.utils.write_basic_config()

num_gpus_on_driver = 1
notebook_launcher(accelerate_train, (f'accelerate_run_{num_gpus_on_driver}_gpu', None), 
                  num_processes=num_gpus_on_driver)

# COMMAND ----------

# DBTITLE 1,Launch with TorchDistributor
from pyspark.ml.torch.distributor import TorchDistributor

def accelerate_train(mlflow_run_name:str='accelerate_run', deepspeed=None):
   
    peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                   mlflow_run_name=mlflow_run_name)
    trainer = train(peft_config, training_arguments, dataset, distributor=True)

    return trainer

# Test this code with TorchDistributor?
num_gpus_per_node = 1
num_nodes = 2
num_processes = num_gpus_per_node * num_nodes

distributor = TorchDistributor(num_processes=num_processes, 
                               local_mode=True, use_gpu=True)
completed_trainer = distributor.run(accelerate_train, f'distributor_run_multinode')

# COMMAND ----------

# DBTITLE 1,Launch with Deepspeed Distributor
# MAGIC %run ./configs/deepspeed_configs

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