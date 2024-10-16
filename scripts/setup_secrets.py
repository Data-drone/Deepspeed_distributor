# Databricks notebook source
# MAGIC %md
# MAGIC # Setting Up Secrets with sdk
# MAGIC
# MAGIC It is never a good idea to have plaintext keys in secrets \
# MAGIC We can use the python sdk to setup secrets

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

# COMMAND ----------

w = WorkspaceClient()
scopes = w.secrets.list_scopes()

# review active scopes
[scope.name for scope in scopes]

# COMMAND ----------

my_scope = 'bootcamp_training'
scrts = w.secrets.list_secrets(scope=my_scope)

# COMMAND ----------

#key_name = f'sdk-{time.time_ns()}'
#scope_name = f'sdk-{time.time_ns()}'

# you can screate scope with this
#w.secrets.create_scope(scope=scope_name)
w.secrets.put_secret(scope=my_scope, key='hf-key', string_value='')

# cleanup
#w.secrets.delete_secret(scope=scope_name, key=key_name)
#w.secrets.delete_scope(scope=scope_name)