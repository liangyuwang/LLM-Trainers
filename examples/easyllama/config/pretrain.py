# train a model
# good for debugging and playing

# data
model_name_or_path = 'NousResearch/Llama-2-7b-hf'
data_path = 'data/wikitext103/'
block_size = 1024

# model
model_name = 'EasyLlama_120M'
pretrained_path = None

# training args
trainer_cls = "pytorch-ddp"
per_device_train_batch_size = 64
per_device_eval_batch_size = 1
gradient_accumulation_steps = 16
learning_rate = 4e-4
weight_decay = 1e-1
adam_beta1 = 0.9
adam_beta2 = 0.95
max_steps = 715256 * 2
lr_scheduler_type = "linear"
warmup_steps = 18    # 2000
output_dir = './out/wikitext103-120m/'
evaluation_strategy = "steps"
eval_steps = 10
logging_dir = './out/wikitext103-120m/'
logging_strategy = "steps"
logging_steps = 10
save_strategy = "steps"
save_steps = 50
save_total_limit = 5
bf16 = False
fp16 = True
half_precision_backend = 'auto'
# disable_tqdm = True
load_best_model_at_end = True
report_to = []
# deepspeed = None
resume_from_checkpoint_path = None
auto_find_batch_size = True
# pipe = True
eval_only = False