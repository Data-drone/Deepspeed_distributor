# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Training with CLI only

# COMMAND ----------

# MAGIC %sh
# MAGIC # to install into the root env
# MAGIC /databricks/python/bin/pip install peft==0.4.0 deepspeed==0.9.4 bitsandbytes==0.39.1 trl==0.5.0

# COMMAND ----------

# setup hf creds
import huggingface_hub
huggingface_key = dbutils.secrets.get(scope='brian-hf', key='hf-key')
huggingface_hub.login(token=huggingface_key)

import os
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ['DATABRICKS_HOST'] = db_host
os.environ['DATABRICKS_TOKEN'] = db_token

# COMMAND ----------

# MAGIC %md
# MAGIC run accelerate config from terminal first

# COMMAND ----------

# MAGIC %sh
# MAGIC accelerate launch train_script.py

