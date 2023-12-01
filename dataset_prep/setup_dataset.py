# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Setup Dataset

# COMMAND ----------

import datasets
# we don't import tokenizer for setting types at the moment
## As loading tokenizer means instantiating cuda which interferes with accelerate

def setup_data(tokenizer, dataset:datasets.Dataset) -> datasets.Dataset:
    """
    This formats the input dataset as per the stanford Alpaca format
    See: https://crfm.stanford.edu/2023/03/13/alpaca.html for more details
    """

    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    PROMPT_NO_INPUT_FORMAT = """{intro}
    
    {instruction_key}
    {instruction}
    
    {response_key}
    {response}
    
    {end_key}""".format(
      intro=INTRO_BLURB,
      instruction_key=INSTRUCTION_KEY,
      instruction="{instruction}",
      response_key=RESPONSE_KEY,
      response="{response}",
      end_key=END_KEY
    )
    
    PROMPT_WITH_INPUT_FORMAT = """{intro}
    
    {instruction_key}
    {instruction}
    
    {input_key}
    {input}
    
    {response_key}
    {response}
    
    {end_key}""".format(
      intro=INTRO_BLURB,
      instruction_key=INSTRUCTION_KEY,
      instruction="{instruction}",
      input_key=INPUT_KEY,
      input="{input}",
      response_key=RESPONSE_KEY,
      response="{response}",
      end_key=END_KEY
    )
    
    def apply_prompt_template(examples):
      instruction = examples["instruction"]
      response = examples["response"]
      context = examples.get("context")
    
      if context:
        full_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
      else:
        full_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
      return { "text": full_prompt }
    
    dataset = dataset.map(apply_prompt_template)

    # need to add in processing to make it tokenized?
    def tokenize_function(allEntries):
      return tokenizer(allEntries['text'], truncation=True, max_length=512,)

    dataset = dataset.map(tokenize_function)

    return dataset