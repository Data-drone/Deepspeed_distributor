# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### HF Low Level loop
# MAGIC This manually manages the train loop

# COMMAND ----------

def full_train_loop(peft_config, training_arguments, dataset, 
                    distributor:bool=True):

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

    model_path = f'/dbfs{model_cache_root}/llama_2_7b'

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

    # Do we still need to set these flags?
    #model.is_parallelizable = False
    #model.model_parallel = False
    ######

    model = get_peft_model(model, peft_config)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = setup_data(tokenizer, dataset)

    # This specific processing is required to make sure it works
    # with the DataLoader and train loop properly
    #train_dataset = train_dataset.remove_columns(['instruction', 'response', 'context', 'text', 'category'])

    # setup trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
       train_dataset,
       collate_fn = data_collator,
       batch_size = training_arguments.per_device_train_batch_size
    )

    try: 
        offload_device = training_arguments.deepspeed['zero_optimization']['offload_optimizer']['device']
        print(offload_device)
    except TypeError:
        offload_device = None

    # if offload_device == 'cpu':
    #    AdamOptimizer = DeepSpeedCPUAdam 
    # else:
    #    AdamOptimizer = FusedAdam
    AdamOptimizer = DeepSpeedCPUAdam

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

    if global_rank == 0:
        active_run = mlflow.start_run(run_name=training_arguments.run_name)

        # Manually log the training_arguments
        mlflow.log_params(training_arguments.to_dict())

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
          mlflow.log_metrics(metrics=run_dict, step=step) if global_rank == 0 else None

    return 'done'










