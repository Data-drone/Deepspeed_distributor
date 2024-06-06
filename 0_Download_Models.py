# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Model Preloading
# MAGIC When testing out a lot of models we don't want to have to keep downloading all the time so we will cache them on dbfs
# MAGIC
# MAGIC To download from Huggingface we need to have a profile and a token. See [Documentation](https://huggingface.co/docs/hub/security-tokens) for details
# MAGIC 
# MAGIC It is never advisable to store passwords and tokens in plain text. In databricks secrets are the way to go. \
# MAGIC See: [AWS Secrets](https://docs.databricks.com/en/security/secrets/index.html) | (Azure Secrets)[https://learn.microsoft.com/en-au/azure/databricks/security/secrets/]
# MAGIC 
# MAGIC Having to setup Databricks cli locally and use terminal may not be possible in some corporate environments you can also use our Python SDK
# MAGIC See our [Python SDK Secrets Examples](https://github.com/databricks/databricks-sdk-py/blob/main/examples/secrets/put_secret_secrets.py) \
# MAGIC **NOTE** with SDK you need to have the newest sdk installed `%pip install -U databricks-sdk`

# COMMAND ----------

# MAGIC %md ## Setting up Secrets
# MAGIC We will use databricks sdk to setup and log a huggingface key. 

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

scope_name = 'finetuning_dev'
key_name = 'hf_key'
hf_key = '<paste your key here DO NOT LOG TO SOURCE CONTROL!>'

# Uncomment to write your key in
# w = WorkspaceClient()
# if scope_name not in [x.name for x in w.secrets.list_scopes()]:
#     w.secrets.create_scope(scope=scope_name)

# w.secrets.put_secret(scope=scope_name, key=key_name, string_value=hf_key)

# COMMAND ----------

# DBTITLE 1,HuggingFace Credential Setup
import huggingface_hub
huggingface_key = dbutils.secrets.get(scope=scope_name, key=key_name)
huggingface_hub.login(token=huggingface_key)

# setup home variables so that we don't run out of cache
# If we don't do this, huggingface can cache to a the root folder `/` which only has ~200GB
# We will set a cache on local_disk0 which is configuratable attached storage
import os
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

# COMMAND ----------

# MAGIC %md ## Download and save models
# MAGIC This will help us to load models faster than redownloading all the time \
# MAGIC We will sav this to a UC volume but it is possible to use dbfs too.

# COMMAND ----------

catalog = 'brian_ml_dev'
schema = 'gen_ai'
volume = 'huggingface_models'

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")
volume_folder = f"/Volumes/{catalog}/{schema}/{volume}/"

# COMMAND ----------

# TODO for the AWQ libs we need to load the safetensors
from huggingface_hub import hf_hub_download, list_repo_files

# repo_list = {'llama_2_7b': 'meta-llama/Llama-2-7b-chat-hf',
#              'llama_2_70b': 'meta-llama/Llama-2-13b-chat-hf'}

repo_list = {'llama_3_8b': 'meta-llama/Meta-Llama-3-8B',
             'llama_3_70b': 'meta-llama/Meta-Llama-3-70B'}

for lib_name in repo_list.keys():
    for name in list_repo_files(repo_list[lib_name]):
        # skip all the safetensors data as we aren't using it and it's time consuming to download
        if "safetensors" in name:
            if lib_name in ['llama_2_13b_awq', 'llama_2_70b_awq']:
                pass
            else:
                continue
        target_path = os.path.join(volume_folder, lib_name, name)
        if not os.path.exists(target_path):
            print(f"Downloading {name}")
            hf_hub_download(
                repo_list[lib_name],
                filename=name,
                local_dir=os.path.join(volume_folder, lib_name),
                local_dir_use_symlinks=False,
            )

# COMMAND ----------
