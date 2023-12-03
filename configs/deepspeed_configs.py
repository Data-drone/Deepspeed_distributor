# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Deepspeed Configurations

# COMMAND ----------

deepspeed_zero_1 = {
	"train_batch_size": "auto",
	"gradient_accumulation_steps": shared_parameters['gradient_accumulation_steps'],
  "gradient_clipping": shared_parameters['gradient_clipping'],
	"train_micro_batch_size_per_gpu": "auto",
	"bf16": {
		"enabled": "auto"
	},
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 500000000,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True,
        "cpu_offload": False
      },
      "optimizer": {
        "type": "AdamW",
        "params": {
          "lr": "auto",
          "betas": [
            0.9,
            0.999
          ],
          "eps": 1e-08
        }
      },
      "scheduler": {
        "type": "WarmupLR",
        "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": "auto",
          "warmup_num_steps": "auto",
          "warmup_type": "linear"
        }
      }
    }

deepspeed_zero_2 = {
    "train_batch_size": "auto",
	  "gradient_accumulation_steps": shared_parameters['gradient_accumulation_steps'],
    "gradient_clipping": shared_parameters['gradient_clipping'],
	  "train_micro_batch_size_per_gpu": "auto",
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
          "lr": "auto",
          "betas": [
            0.9,
            0.999
          ],
          "eps": 1e-08
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": "auto",
          "warmup_num_steps": "auto",
          "warmup_type": "linear"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto"
    }
}

deepspeed_zero_3 = {
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
    "steps_per_print": 10,
    "train_micro_batch_size_per_gpu": shared_parameters['per_device_batch_size'],
    "train_batch_size": "auto",
    "wall_clock_breakdown": True
}

deepspeed_zero_3_offload = {
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
    "steps_per_print": 10,
    "train_micro_batch_size_per_gpu": shared_parameters['per_device_batch_size'],
    "train_batch_size": "auto",
    "wall_clock_breakdown": True
}