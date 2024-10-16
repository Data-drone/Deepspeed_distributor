# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### HF Low Level loop
# MAGIC This manually manages the train loop

# COMMAND ----------

def full_train_loop(peft_config, training_arguments, dataset, 
                    distributor:bool=True, mlflow_parent_run=None):

    """
    Deepspeed isn't handling train_batch here if it is string:
    https://github.com/microsoft/DeepSpeed/blob/f57fc4c95a6a5194757b57704f60f009dde25680/deepspeed/runtime/config.py#L903
    guessing we need to specify a batch size which means we need to calculate it all first
    """

    import os
    import mlflow

    import torch
    from torch.utils.data import DataLoader
    
    import deepspeed
    from deepspeed import get_accelerator
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

    from transformers import (
       AutoModelForCausalLM, AutoTokenizer,
       DataCollatorForLanguageModeling
    )
    from peft import get_peft_model
    from deepspeed.utils import logger as ds_logger

    os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_path
    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
    #os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = 'True'
    
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token

    if distributor:
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'

    mlflow.set_registry_uri('databricks')

    model_path = f'{model_cache_root}/llama_3_1_8b/'

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    device = torch.device(get_accelerator().device_name())
    global_rank = torch.distributed.get_rank()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        #device_map = {"":int(local_rank)},
        torch_dtype=torch.bfloat16,
        cache_dir=model_path,
        local_files_only=True,
        low_cpu_mem_usage=False
    )

    model = get_peft_model(model, peft_config)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = setup_data(tokenizer, dataset)

    # removve string columns
    train_dataset = train_dataset.remove_columns(['text', 'category', 'instruction', 
                                                 'context', 'response'])

    # setup trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
       train_dataset,
       collate_fn = data_collator,
       batch_size = training_arguments.per_device_train_batch_size
    )

    deepspeed_arg = training_arguments.deepspeed
    ds_logger.info(f'deepspeed argument is of type: {type(deepspeed_arg)}')

    # Deepspeed Args can be a dict or a string
    ## When it is a string we need to load the file first into a dict
    if type(deepspeed_arg) == str:
        import json
        with open(training_arguments.deepspeed, 'r') as file:
            deepspeed_config_load = json.load(file)

    elif type(deepspeed_arg) == dict:
        deepspeed_config_load = deepspeed_arg

    try: 
        offload_device = deepspeed_config_load['zero_optimization']['offload_optimizer']['device']
        ds_logger.info(f'DeepSpeed Offload: {offload_device}')
    except (TypeError, KeyError) as e:
        ds_logger.info(f'Offload detection error: {e}')
        offload_device = None

    # We need different optimizer depending on whether it is using offload or not
    if offload_device == 'cpu':
       AdamOptimizer = DeepSpeedCPUAdam 
    else:
       AdamOptimizer = FusedAdam
    
    optimizer = AdamOptimizer(model.parameters(),
                              lr=training_arguments.learning_rate,
                              betas=(0.9, 0.95))
 
    # model, optimizer
    initialised_var  = deepspeed.initialize(
       model = model,
       optimizer = optimizer,
       dist_init_required=False,
       config = training_arguments.deepspeed
    )

    model = initialised_var[0]
    optimizer = initialised_var[1]

    # with manual loop we will have to add manual mlflow
    # variables:
    # training_arguments, model, train_dataloader
    # device 

    ## setup distributed mlflow system metric logging
    ## We want to log system usage on all nodes so we need to make sure that they are all
    ## nested back with the correct parent

    if mlflow_parent_run:
        from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID  
        run_tags = {MLFLOW_PARENT_RUN_ID: mlflow_parent_run.info.run_id}
    else:
        run_tags = {}

    ds_logger.info(f'run_tags are: {run_tags}')

    # We want to log all configs and track loss on our primary node
    # But not on the other mlflow runs that exist just to log system stats
    if global_rank == 0:
        active_run = mlflow.start_run(run_name=training_arguments.run_name,
                                      tags=run_tags)

        # Manually log the training_arguments
        mlflow.log_params(training_arguments.to_dict())

        ## Deepspeed config needs to be unpacked separately
        ## some DS variables overlap with HF ones
        mod_ds_args = {"ds_" + key: value for key, value in deepspeed_config_load.items()}
        mlflow.log_params(mod_ds_args)
    
    else:
        active_run = mlflow.start_run(run_name=f"{training_arguments.run_name}_rank_{global_rank}",
                                      tags=run_tags)


    # Now we can start the run loop
    for epoch in range(training_arguments.num_train_epochs):
      model.train()

      for step, batch in enumerate(train_dataloader):
          batch.to(device)
          outputs = model(**batch, use_cache=False)

          loss = outputs.loss

          model.backward(loss)
          model.step()

          run_dict = {
              'train_loss': loss,
              'step': step
            }

          # we need to make sure step is defined properly
          # We also only log loss on rank 0 node
          if global_rank == 0:
            mlflow.log_metrics(metrics=run_dict, step=step) if global_rank == 0 else None

    return 'done'










