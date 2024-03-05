datasets_info = [
    # The following datasets are for pre-training
    # {
    #     'dataset_path': 'openwebtext',
    #     'dataset_name': None,
    #     'text_column': 'text',
    #     'train_name': 'train',
    #     'val_name': None,
    #     'test_name': None,
    #     'split_ratio': 0.01,  # try 0.0001
    # },
    # {
    #     'dataset_path': 'wikitext',
    #     'dataset_name': 'wikitext-103-raw-v1',
    #     'text_column': 'text',
    #     'train_name': 'train',
    #     'val_name': 'validation',
    #     'test_name': 'test',
    #     'split_ratio': None,
    # },
    # {
    #     'dataset_path': 'cerebras/SlimPajama-627B',
    #     'dataset_name': None,
    #     'text_column': 'text',
    #     'train_name': 'train',
    #     'val_name': 'validation',
    #     'test_name': 'test',
    #     'split_ratio': None,
    # },

    # The following datasets are for instruction-tuning
    # {
    #     'dataset_path': 'tatsu-lab/alpaca',
    #     'dataset_name': None,
    #     'instruction_column': 'text',
    #     'input_column': 'input',
    #     'output_column': 'output',
    #     'train_name': 'train',
    #     'val_name': None,
    #     'test_name': None,
    #     'split_ratio': 0.01,
    # }

    # The following datasets are for debugging or fine-tuning
    {
        'dataset_path': 'ptb_text_only',
        'dataset_name': None,
        'text_column': 'sentence',
        'train_name': 'train',
        'val_name': 'validation',
        'test_name': 'test',
        'split_ratio': None,
    },
    # {
    #     'dataset_path': 'heliosbrahma/mental_health_chatbot_dataset',
    #     'dataset_name': None,
    #     'text_column': 'text',
    #     'train_name': 'train',
    #     'val_name': None,
    #     'test_name': None,
    #     'split_ratio': 0.1,
    # },
    # {
    #     'dataset_path': 'wikitext',
    #     'dataset_name': 'wikitext-2-v1',
    #     'text_column': 'text',
    #     'train_name': 'train',
    #     'val_name': 'validation',
    #     'test_name': 'test',
    #     'split_ratio': None,
    # },
]
model_name_or_path = 'NousResearch/Llama-2-7b-hf'
data_path = 'data/ptb_text_only/'
block_size = 1024