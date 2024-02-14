# fine-tune a model with instruction datasets
# good for debugging and playing

# data
model_name_or_path = 'NousResearch/Llama-2-7b-hf'
data_path = 'data/alpaca/'
block_size = 1024

# model
model_name = 'EasyLlama_120M'
pretrained_path = None

# training args
per_device_train_batch_size = 64
per_device_eval_batch_size = 1
gradient_accumulation_steps = 16
learning_rate = 2e-5
weight_decay = 1e-1
adam_beta1 = 0.9
adam_beta2 = 0.95
max_steps = 10
lr_scheduler_type = "linear"
warmup_steps = 1
output_dir = './out/alpaca-chat-120m/'
evaluation_strategy = "steps"
eval_steps = 2
logging_dir = './out/alpaca-chat-120m/'
logging_strategy = "steps"
logging_steps = 2
save_strategy = "steps"
save_steps = 2
save_total_limit = 5
bf16 = False
fp16 = True
half_precision_backend = "auto"
# disable_tqdm = True
load_best_model_at_end = True
report_to = []
# deepspeed = None
resume_from_checkpoint_path = 'out/alpaca-chat-120m/step-3100/ckpt.pt'
auto_find_batch_size = True
eval_only = False