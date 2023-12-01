# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### HF Trainer train loop
# MAGIC This utilises hf trainer
# MAGIC as of 4.34 it works with ZeRo 1 and 2 but not 3

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

    model_path = f'/dbfs{model_cache_root}/llama_2_7b'

    device_map_var = None if deepspeed else {"":int(os.environ.get("LOCAL_RANK"))}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = device_map_var,
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
    )

    trainer.train()

    # if we return the trainer object we get an error
    return 'done'