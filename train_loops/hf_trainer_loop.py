# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### HF Trainer train loop
# MAGIC This utilises hf trainer
# MAGIC fixed with deepspeed 0.12.3

# COMMAND ----------

def train(peft_config, training_arguments, dataset, distributor=True, deepspeed=False):

    # This is our main training function
    # we do our inports inside so that we don't initiate cuda for the native accelerator code

    from transformers import (
       AutoModelForCausalLM, AutoTokenizer,
       DataCollatorForLanguageModeling, Trainer
    )
    from peft import get_peft_model

    import mlflow
    import torch
    import os
    import logging
    import sys

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                      "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
    #os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = 'True'
    
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    if distributor:
      os.environ['NCCL_IB_DISABLE'] = '1'
      os.environ['NCCL_P2P_DISABLE'] = '1'

    mlflow.set_registry_uri('databricks')

    model_path = f'/dbfs{model_cache_root}/llama_2_7b'

    logger.info(f'Deepspeed: {deepspeed}')
    logger.info(f'Deepspeed Settings: {training_arguments.deepspeed}')

    device_map_var = None if deepspeed else {"":int(os.environ.get("LOCAL_RANK"))}
    low_mem_usage = False if deepspeed else True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device_map_var,
        torch_dtype=torch.bfloat16,
        cache_dir=model_path,
        local_files_only=True,
        low_cpu_mem_usage=low_mem_usage
    )

    # Do we still need to set these flags?
    model.is_parallelizable = False
    model.model_parallel = False

    # if you want to do full parameter pretraining just comment out this command
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
    )

    trainer.train()

    # if we return the trainer object we get an error
    return 'done'