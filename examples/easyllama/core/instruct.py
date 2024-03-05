import os
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import torch
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM, 
    LlamaConfig, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))
from data.utils import load_dataset, format_row_as_instruction_prompt
from config.model import ModelConfigs


exec(open('utils/configurator.py').read()) # overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, type(None)))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
def load_args(args, config: dict):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

if hasattr(config, 'deepspeed'):
    import deepspeed
    deepspeed.init_distributed()

# Define dataset:
tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
dataset = load_dataset(config['data_path'])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Define the model
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
if hasattr(config, 'pretrained_path'):
    model = LlamaForCausalLM.from_pretrained(config['pretrained_path'], device_map=device_map)
else:
    model_config = LlamaConfig()
    load_args(model_config, ModelConfigs[config['model_name']])
    model = LlamaForCausalLM(config=model_config)
    model.device_map = device_map


# Define the trainer
training_args = TrainingArguments(output_dir=config['output_dir'])
load_args(training_args, config)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    max_seq_length=config['block_size'],
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_row_as_instruction_prompt,
)


# Train and eval the model
if config['eval_only']:
    results = trainer.evaluate(dataset['test'])
    print("Original Loss: ", results["eval_loss"])
    print("Evaluation perplexity: ", np.exp(results["eval_loss"]))
    exit()

trainer.train()

results = trainer.evaluate(dataset['test'])
print("Original Loss: ", results["eval_loss"])
print("Evaluation perplexity: ", np.exp(results["eval_loss"]))

