#### Testing script
import mlflow
import torch

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments

from peft import LoraConfig

from trl import SFTTrainer

from accelerate import Accelerator


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


if __name__ == '__main__':

    mlflow.set_experiment('/Users/brian.law@databricks.com/deepspeed-distributor')

    accelerator = Accelerator()

    dataset_name = "databricks/databricks-dolly-15k"
    dataset = load_dataset(dataset_name, split="train")

    dataset = dataset.map(apply_prompt_template)

    model = "meta-llama/Llama-2-7b-hf"
    revision = "351b2c357c69b4779bde72c0e7f7da639443d904"

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        revision=revision,
        #device_map={'':torch.cuda.current_device()},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    #model = model.to(accelerator.device)

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'] # Choose all linear layers from the model
    )

    output_dir = "/local_disk0/results"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 500
    logging_steps = 100
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 1000
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        bf16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        ddp_find_unused_parameters=False,
    )

    max_seq_length = 512

    #dataset, model = accelerator.prepare(dataset, model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()