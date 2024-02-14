import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer


exec(open('configurator.py').read()) # overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, type(None)))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'])
if 'Llama' in config['model_name_or_path']:
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
else:
    tokenizer.pad_token = tokenizer.eos_token

# rename instruction dataset columns
def rename_columns(example):
    example["instruction"] = example.pop(dataset_info["instruction_column"])
    example["input"] = example.pop(dataset_info["input_column"])
    example["output"] = example.pop(dataset_info["output_column"])
    return example


# process multi datasets
new_train_datasets, new_val_datasets, new_test_datasets = [], [], []
for dataset_info in config['datasets_info']:
    print(f"Tokenizing {dataset_info['dataset_path']}...")

    dataset = load_dataset(path=dataset_info['dataset_path'], name=dataset_info['dataset_name'])
    # Split dataset if necessary
    if dataset_info['val_name'] is None:
        if dataset_info['split_ratio'] is not None:
            print(f"Validation set is not provided. Splitting {dataset_info['train_name']} dataset with ratio {dataset_info['split_ratio']}.")
            dataset = dataset[dataset_info['train_name']].train_test_split(test_size=dataset_info['split_ratio'])
            dataset['validation'] = dataset['test']
            del dataset['test']
        else:
            raise ValueError("If val_name is None, the split_ratio must be provided.")
    if dataset_info['test_name'] is None:
        print(f"Test set is not provided. Using {dataset_info['val_name']} as test set.")
        dataset['test'] = dataset['validation'] if dataset_info['val_name'] is None else dataset[dataset_info['val_name']]

    # Rename keys
    new_dataset = DatasetDict({
        'train': dataset[dataset_info['train_name']],
        'validation': dataset[dataset_info['val_name']] if dataset_info['val_name'] is not None else dataset['validation'],
        'test': dataset[dataset_info['test_name']] if dataset_info['test_name'] is not None else dataset['test']
    })

    # filter dataset
    new_dataset = new_dataset.filter(
        lambda example: (
            len(example[dataset_info["instruction_column"]]) + len(example[dataset_info["input_column"]]) + len(example[dataset_info["output_column"]])
        ) <= config['block_size'])
    
    # rename columns
    new_dataset = new_dataset.map(rename_columns)

    new_train_datasets.append(new_dataset['train'])
    new_val_datasets.append(new_dataset['validation'])
    new_test_datasets.append(new_dataset['test'])

# save to disk
new_train_datasets = concatenate_datasets(new_train_datasets).shuffle()
new_val_datasets = concatenate_datasets(new_val_datasets).shuffle()
new_test_datasets = concatenate_datasets(new_test_datasets).shuffle()
final_new_dataset = DatasetDict({
    'train': new_train_datasets,
    'validation': new_val_datasets,
    'test': new_test_datasets
})
final_new_dataset.save_to_disk(config['data_path'])

