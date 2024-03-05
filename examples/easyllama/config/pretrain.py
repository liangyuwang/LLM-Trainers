# train a model
# good for debugging and playing
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# data
model_name_or_path = 'NousResearch/Llama-2-7b-hf'
data_path = 'data/ptb_text_only/'
block_size = 1024

# model
model_name = 'EasyLlama_120M'
pretrained_path = None

# training args
trainer_cls = "pytorch-fsdp"
per_device_train_batch_size = 1
per_device_eval_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 4e-4
weight_decay = 1e-1
adam_beta1 = 0.9
adam_beta2 = 0.95
max_steps = 715256 * 2
lr_scheduler_type = "linear"
warmup_steps = 1    # 2000
output_dir = './out/ptb_text_only-120m/'
overwrite_output_dir = True
evaluation_strategy = "steps"
eval_steps = 10
logging_dir = './out/ptb_text_only-120m/'
logging_strategy = "steps"
logging_steps = 10
save_strategy = "steps"
save_steps = 5
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
pipe = True
pipe_config = {
    'mode': 'layers',
    'layers_name': 'model.layers',
    'all_modules': ["model.embed_tokens", "model.layers", "model.norm", "lm_head"],
    'splits': 4,
    'chunks': 1,
}
fsdp = True
fsdp_config = {
    'min_num_params': 0,
    'mixed_precision_type': "fsdp_bf16",
    'fsdp_activation_checkpointing': True,
    'limit_all_gathers': False,
    'sharding_strategy': ShardingStrategy.FULL_SHARD, #HYBRID_SHARD, SHARD_GRAD_OP
    'checkpoint_type': StateDictType.FULL_STATE_DICT, # alternatively can use SHARDED_STATE_DICT to avoid OOMs
}
eval_only = False