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

block_size = config['block_size']

def tokenize_function(examples):
    return tokenizer(examples[dataset_info['text_column']])

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size]
            for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# process multi datasets
tokenized_train_datasets, tokenized_val_datasets, tokenized_test_datasets = [], [], []
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

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        num_proc=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        remove_columns=[dataset_info['text_column']],
    )

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )
    tokenized_train_datasets.append(tokenized_dataset['train'])
    tokenized_val_datasets.append(tokenized_dataset['validation'])
    tokenized_test_datasets.append(tokenized_dataset['test'])

# save to disk
tokenized_train_datasets = concatenate_datasets(tokenized_train_datasets).shuffle()
tokenized_val_datasets = concatenate_datasets(tokenized_val_datasets).shuffle()
tokenized_test_datasets = concatenate_datasets(tokenized_test_datasets).shuffle()
final_tokenized_dataset = DatasetDict({
    'train': tokenized_train_datasets,
    'validation': tokenized_val_datasets,
    'test': tokenized_test_datasets
})
final_tokenized_dataset.save_to_disk(config['data_path'])

