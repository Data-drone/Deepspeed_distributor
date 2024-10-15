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

# DBTITLE 1,HuggingFace Credential Setup
import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='bootcamp_training', key='hf-key')
huggingface_hub.login(token=huggingface_key)

# setup home variables so that we don't run out of cache
# If we don't do this, huggingface can cache to a the root folder `/` which only has ~200GB
# We will set a cache on local_disk0 which is configuratable attached storage
import os
os.environ['HF_HOME'] = '/local_disk0/hf'
os.environ['HUGGING_FACE_HUB_TOKEN'] = huggingface_key

# COMMAND ----------

# setup model path
# In an enterprise env you may setup a shared folder for all users but I will store in a user folder here
username = spark.sql("SELECT current_user()").first()['current_user()']

# migrate to volumes
catalog = 'brian_ml_dev'
schema = 'deepspeed_distributor'
volume = 'model_Weights'

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

#downloads_home = f'/home/{username}/hf_models'
#dbutils.fs.mkdirs(downloads_home)
#dbfs_downloads_home = f'/dbfs{downloads_home}'
volume_downloads_home = f'/Volumes/{catalog}/{schema}/{volume}'

# COMMAND ----------

# TODO for the AWQ libs we need to load the safetensors
from huggingface_hub import hf_hub_download, list_repo_files

repo_list = {'llama_3_1_8b': 'meta-llama/Llama-3.1-8B'}

for lib_name in repo_list.keys():
    for name in list_repo_files(repo_list[lib_name]):
        target_path = os.path.join(volume_downloads_home, lib_name, name)
        if not os.path.exists(target_path):
            print(f"Downloading {name}")
            hf_hub_download(
                repo_list[lib_name],
                filename=name,
                local_dir=os.path.join(volume_downloads_home, lib_name),
                local_dir_use_symlinks=False,
            )

# COMMAND ----------
