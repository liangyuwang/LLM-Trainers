import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM, 
    LlamaConfig, 
    DataCollatorForLanguageModeling
)
import llmtrainers
from llmtrainers.pytorch import TrainingArguments

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))
from data.utils import load_dataset
from config.model import ModelConfigs


exec(open('utils/configurator.py').read()) # overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, dict, type(None)))]
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
tokenized_datasets = load_dataset(config['data_path'])
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
Trainer = llmtrainers.trainer_cls(select=config['trainer_cls'])
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)


# Train and eval the model
if config['eval_only']:
    results = trainer.evaluate(tokenized_datasets['test'])
    print("Original Loss: ", results["eval_loss"])
    print("Evaluation perplexity: ", np.exp(results["eval_loss"]))
    exit()

trainer.train(resume_from_checkpoint=config['resume_from_checkpoint_path'])

results = trainer.evaluate(tokenized_datasets['test'])
print("Original Loss: ", results["eval_loss"])
print("Evaluation perplexity: ", np.exp(results["eval_loss"]))

