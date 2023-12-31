# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Training and Finetuning on Huggingface w ZeRO and deepspeed
# MAGIC
# MAGIC Lets see how we can scale the native library functions \
# MAGIC But this is how we can leverage Accelerate driven HF Trainer on Databricks
# MAGIC 
# MAGIC *Notes*
# MAGIC This code was tested on MLR 14.2 \
# MAGIC Versions of transformers and deepspeed are quite important \

# COMMAND ----------

# MAGIC %pip install peft==0.6.0 deepspeed==0.12.3 bitsandbytes==0.41.1 mlflow==2.9.2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# we can create a generic setup
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Environment Configurations
# MAGIC To start with, we will setup some basic configurations \
# MAGIC We log the notebook token and host so that we can connect to the mlflow service correctly \
# MAGIC We will also manually setup an mlflow experiment \
# MAGIC The cache directories we setup to make sure that models are datasets are use dbfs object store paths so that all data and models are persisted \
# MAGIC This notebook is setup so that you can try out distributed deep learning with:
# MAGIC - Native Accelerate (single node only)
# MAGIC - Spark TorchDistributor (wo DeepSpeed Acceleration)
# MAGIC - Spark DeepspeedDistributor (w DeepSpeed using HF Trainer)
# MAGIC - Spark TorchDistributor (w DeepSpeed using a custom training loop)

# COMMAND ----------

# we setup all these widgets so that you can setup a databricks Workflow and queue these up to execute without having to manually watch the job
dbutils.widgets.dropdown("distribution_mechanism", "Deepspeed_HF_Trainer", 
                         ["accelerate", "TorchDistributor", "Deepspeed_HF_Trainer", "Deepspeed_Custom_Loop"])
dbutils.widgets.text("num_gpus_per_node", "2")
dbutils.widgets.text("num_nodes", "1")
dbutils.widgets.dropdown("deepspeed_stage", "stage_3_offload", ["stage_1", "stage_2", "stage_3", "stage_3_offload"])

# COMMAND ----------

# Databricks configuration and MLflow setup
browser_host = spark.conf.get("spark.databricks.workspaceUrl")
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# MLflow configuration
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/deepspeed-distributor'

datasets_cache = f'/home/{username}/datasets_cache'
model_cache_root = f'/home/{username}/hf_models'
dbfs_datasets_cache = f'/dbfs{datasets_cache}'

# execution option
exec_opt = dbutils.widgets.get("distribution_mechanism")
num_gpus = int(dbutils.widgets.get("num_gpus_per_node"))
num_nodes = int(dbutils.widgets.get("num_nodes"))
deepspeed_stage = dbutils.widgets.get("deepspeed_stage")

# COMMAND ----------

# MAGIC %md
# MAGIC # Training Parameter Configurations
# MAGIC When using HuggingFace together with Deepspeed we need to make sure certain parameters are correctly aligned. \
# MAGIC Otherwise the training code will not know whether to use the HF configured `batch_size` or deepspeed one for example. \
# MAGIC
# MAGIC The way we will ensure this is to setup a shared parameter dictionary that both the HF Training Arguments and DeepSpeed configuration will read from.
# COMMAND ----------

shared_parameters = {
   "gradient_accumulation_steps": 1,
   "gradient_clipping": 0.3,
   "per_device_batch_size": 4,
   "learning_rate": 2e-4,
   "warmup_steps": 100
}

# COMMAND ----------

def setup_params(shared_parameters:dict, mlflow_run_name: str='single_run', deepspeed_config=None):

    # setup training arguments
    # We do this inside a function so that we don't initialise cuda before Accelerate takes over
    # this is needed for `Accelerate` notebook launcher to work.  

    # the PEFT calls for LoRa training are only used if we do LoRa training
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
    lr_scheduler_type = "constant"

    # this is the HF TrainingArguments object which is needed for the HF Trainer
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
        warmup_steps = shared_parameters['warmup_steps'],
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        ddp_find_unused_parameters=False,
        run_name=mlflow_run_name,
        deepspeed=deepspeed_config
    )

    return peft_config, training_arguments

# COMMAND ----------

# DBTITLE 1,Load Dataset

# We will use the databricks dolly dataset See the following blog for more details
# https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm 
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

# Note that rerunning this notebook without restarting the node may cause issues
# This is because accelerate notebook_launcher requires that CUDA status be fresh when it starts
# Importing `torch`` or `transformers` for example will initialise CUDA stopping `notebook_launcher``

if exec_opt == 'accelerate':
    import accelerate
    from accelerate import notebook_launcher

    def accelerate_train(mlflow_run_name:str='accelerate_run', deepspeed=None):

        peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                    mlflow_run_name=mlflow_run_name)
        trainer = train(peft_config, training_arguments, dataset, distributor=True)

        return trainer

    # if we need to write a config
    # accelerate.utils.write_basic_config()

    num_gpus_on_driver = num_gpus

    logger.info(f"Launching job with accelerate with {num_gpus_on_driver} gpus")
    notebook_launcher(accelerate_train, (f'accelerate_run_{num_gpus_on_driver}_gpu', None), 
                    num_processes=num_gpus_on_driver)

# COMMAND ----------

# DBTITLE 1,Launch with TorchDistributor
    
# To distribute across multinode, we need at least TorchDistributor

if exec_opt == 'TorchDistributor':  
    from pyspark.ml.torch.distributor import TorchDistributor

    def accelerate_train(mlflow_run_name:str='accelerate_run', deepspeed=None):
    
        peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                    mlflow_run_name=mlflow_run_name)
        trainer = train(peft_config, training_arguments, dataset, distributor=True)

        return trainer

    num_gpus_per_node = num_gpus
    num_nodes = num_nodes
    num_processes = num_gpus_per_node * num_nodes
    local_status = True if num_nodes == 1 else False


    distributor = TorchDistributor(num_processes=num_processes, 
                                local_mode=local_status, use_gpu=True)
    
    logger.info(f"Launching job with TorchDistributor with {num_gpus_per_node} gpus per node and {num_nodes} nodes")
    completed_trainer = distributor.run(accelerate_train, f'distributor_run_multinode')

# COMMAND ----------

# MAGIC %md
# MAGIC # Using the DeepSpeed Distributor
# MAGIC Deepspeed requires setting up the deepspeed configurations as discussed earlier.\
# MAGIC It gets more complicated when we use it with HuggingFace Trainer because we need to make sure that parameters which appear in both are aligned \
# MAGIC That is why we created the shared parameter dictionary earlier and use it instantiate our deepspeed configs \
# MAGIC In your own production code you may want to just setting for one deepspeed configuration and do away with this complexity.
    
# COMMAND ----------

# DBTITLE 1,Deepspeed Configurations
# MAGIC %run ./configs/deepspeed_configs

# COMMAND ----------

# DBTITLE 1,Mapping widgets value to configs
if deepspeed_stage == 'stage_1':    
    deepspeed_mode = deepspeed_zero_1
elif deepspeed_stage == 'stage_2':
    deepspeed_mode = deepspeed_zero_2
elif deepspeed_stage == 'stage_3':
    deepspeed_mode = deepspeed_zero_3
elif deepspeed_stage == 'stage_3_offload':
    deepspeed_mode = deepspeed_zero_3_offload

# COMMAND ----------

# DBTITLE 1, Low Level Loop if you want to customise it
# MAGIC %run ./train_loops/deepspeed_manual_loop

# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

def parse_deepspeed():

    """
    Spark DeepSpeed Distributor adds in the deepspeed configurations params
    As part of the initialisation process
    To make use of them in our training process, we can parse the args to use in our train loops
    We can receive and use them with argparse
    """

    import argparse
    import deepspeed

    parser = argparse.ArgumentParser(description='Torch Distributor Training')
    parser = deepspeed.add_config_arguments(parser)

    return parser

# COMMAND ----------

# DBTITLE 1, HF Trainer
    
if exec_opt == 'Deepspeed_HF_Trainer':  
    
    def deepspeed_train():
        
        parsed_args = parse_deepspeed().parse_args()
        print(parsed_args)
    
        peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                    mlflow_run_name='deepspeed_distributor_w_trainer',
                                                    deepspeed_config=parsed_args.deepspeed_config)
        # can use `train` too
        trainer = train(peft_config, training_arguments, dataset, 
                        distributor=True,
                        deepspeed=parsed_args.deepspeed)

        return trainer

    num_gpus = num_gpus
    num_nodes = num_nodes
    num_processes = num_gpus * num_nodes
    local_status = True if num_nodes == 1 else False

    deepspeed_dict = deepspeed_mode

    deepspeed_dict['train_batch_size'] = (
        shared_parameters["per_device_batch_size"] *
        shared_parameters["gradient_accumulation_steps"] *
        num_processes
    )

    distributor = DeepspeedTorchDistributor(numGpus=num_gpus, nnodes=num_nodes, localMode=local_status, 
                                            useGpu=True, deepspeedConfig = deepspeed_dict)

    completed_trainer = distributor.run(deepspeed_train)

# COMMAND ----------

# DBTITLE 1, Low Level Loop
    
if exec_opt == 'Deepspeed_Custom_Loop':  

    import mlflow
    
    def deepspeed_train(parent_run: str):
        
        parsed_args = parse_deepspeed().parse_args()
        print(parsed_args)
    
        peft_config, training_arguments = setup_params(shared_parameters=shared_parameters,
                                                    mlflow_run_name='deepspeed_distributor_w_config_low_level',
                                                    deepspeed_config=parsed_args.deepspeed_config)
        # can use `train` too

        trainer = full_train_loop(peft_config, training_arguments, dataset, 
                        distributor=True, mlflow_parent_run=parent_run)

        return trainer

    num_gpus = num_gpus
    num_nodes = num_nodes
    num_processes = num_gpus * num_nodes
    local_status = True if num_nodes == 1 else False

    deepspeed_dict = deepspeed_mode

    deepspeed_dict['train_batch_size'] = (
        shared_parameters["per_device_batch_size"] *
        shared_parameters["gradient_accumulation_steps"] *
        num_processes
    )

    if num_nodes > 1:
        mlflow.set_experiment(experiment_path)
        parent_run = mlflow.start_run(
            run_name='deepspeed_distributor_w_config_low_level')
    else:
        parent_run = None

    distributor = DeepspeedTorchDistributor(numGpus=num_gpus, nnodes=num_nodes, localMode=local_status, 
                                            useGpu=True, deepspeedConfig = deepspeed_dict)

    completed_trainer = distributor.run(deepspeed_train, parent_run)

# COMMAND ----------