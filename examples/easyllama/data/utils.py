from datasets import load_from_disk, concatenate_datasets, DatasetDict


def load_dataset(data_path):
    if isinstance(data_path, str):
        tokenized_datasets = load_from_disk(data_path)
    elif isinstance(data_path, list):
        dataset_paths = data_path
        train_datasets, val_datasets, test_datasets = [], [], []
        # Load all datasets
        for path in dataset_paths:
            print(f"Loading dataset from {path}")
            dataset = load_from_disk(path)
            if 'train' in dataset:
                train_datasets.append(dataset['train'])
            if 'validation' in dataset:
                val_datasets.append(dataset['validation'])
            if 'test' in dataset:
                test_datasets.append(dataset['test'])
        # Concatenate all loaded datasets
        concatenated_train_dataset = concatenate_datasets(train_datasets).shuffle() if train_datasets else None
        concatenated_val_dataset = concatenate_datasets(val_datasets).shuffle() if val_datasets else None
        concatenated_test_dataset = concatenate_datasets(test_datasets).shuffle() if test_datasets else None
        # Combine into a DatasetDict
        tokenized_datasets = DatasetDict({
            'train': concatenated_train_dataset,
            'validation': concatenated_val_dataset,
            'test': concatenated_test_dataset
        })
    else:
        raise ValueError(f"Invalid data_path: {data_path}")
    return tokenized_datasets


def format_row_as_instruction_prompt(example):
    """
        See tutorial https://deci.ai/blog/instruction-tuning-of-decilm-using-lora/
    """
    has_input = example.get('input', None) is not None
    if has_input:
        primer_prompt = ("Below is an instruction that describes a task, paired with an input "
                         "that provides further context. Write a response that appropriately completes the request.")
        input_template = f"### Input: \n{example['input']}\n\n"
    else:
        primer_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.")
        input_template = ""
    instruction_template = f"### Instruction: \n{example['instruction']}\n\n"
    # Check if 'output' key exists
    if example.get('output', None):
        response_template = f"### Response: \n{example['output']}\n\n"
    else:
        response_template = ""
    return f"{primer_prompt}\n\n{instruction_template}{input_template}{response_template}"

