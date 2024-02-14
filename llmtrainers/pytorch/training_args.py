from typing import Optional, Dict
from dataclasses import field
import io
import json
import warnings
from enum import Enum

import transformers


class TrainingArguments(transformers.TrainingArguments):
    """
        Add args:
        pipe (`bool`, `str` or list of [`~trainer_utils.FSDPOption`], *optional*, defaults to `''`):
            Use PyTorch Distributed Parallel Training (in distributed training only).
        pipe_config (`str` or `dict`, *optional*):
            Config to be used with pippy (Pytorch Distributed Parallel Training).

        fsdp (`bool`, `str` or list of [`~trainer_utils.FSDPOption`], *optional*, defaults to `''`):
            Use PyTorch Distributed Parallel Training (in distributed training only).

            A list of options along the following:

            - `"full_shard"`: Shard parameters, gradients and optimizer states.
            - `"shard_grad_op"`: Shard optimizer states and gradients.
            - `"hybrid_shard"`: Apply `FULL_SHARD` within a node, and replicate parameters across nodes.
            - `"hybrid_shard_zero2"`: Apply `SHARD_GRAD_OP` within a node, and replicate parameters across nodes.
            - `"offload"`: Offload parameters and gradients to CPUs (only compatible with `"full_shard"` and
              `"shard_grad_op"`).
            - `"auto_wrap"`: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.
        fsdp_config (`str` or `dict`, *optional*):
            Config to be used with fsdp (Pytorch Distributed Parallel Training). The value is either a location of
            fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.

            A List of config and its options:
                - min_num_params (`int`, *optional*, defaults to `0`):
                    FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is
                    passed).
                - transformer_layer_cls_to_wrap (`List[str]`, *optional*):
                    List of transformer layer class names (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`,
                    `T5Block` .... (useful only when `fsdp` flag is passed).
                - backward_prefetch (`str`, *optional*)
                    FSDP's backward prefetch mode. Controls when to prefetch next set of parameters (useful only when
                    `fsdp` field is passed).

                    A list of options along the following:

                    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter's
                      gradient
                        computation.
                    - `"backward_post"` : This prefetches the next set of parameters after the current set of
                      parameter’s
                        gradient computation.
                - forward_prefetch (`bool`, *optional*, defaults to `False`)
                    FSDP's forward prefetch mode (useful only when `fsdp` field is passed).
                     If `"True"`, then FSDP explicitly prefetches the next upcoming all-gather while executing in the
                     forward pass.
                - limit_all_gathers (`bool`, *optional*, defaults to `False`)
                    FSDP's limit_all_gathers (useful only when `fsdp` field is passed).
                     If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight
                     all-gathers.
                - use_orig_params (`bool`, *optional*, defaults to `True`)
                    If `"True"`, allows non-uniform `requires_grad` during init, which means support for interspersed
                    frozen and trainable paramteres. Useful in cases such as parameter-efficient fine-tuning. Please
                    refer this
                    [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019
                - sync_module_states (`bool`, *optional*, defaults to `True`)
                    If `"True"`, each individually wrapped FSDP unit will broadcast module parameters from rank 0 to
                    ensure they are the same across all ranks after initialization
                - activation_checkpointing (`bool`, *optional*, defaults to `False`):
                    If `"True"`, activation checkpointing is a technique to reduce memory usage by clearing activations of
                    certain layers and recomputing them during a backward pass. Effectively, this trades extra
                    computation time for reduced memory usage.
                - xla (`bool`, *optional*, defaults to `False`):
                    Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature
                    and its API may evolve in the future.
                - xla_fsdp_settings (`dict`, *optional*)
                    The value is a dictionary which stores the XLA FSDP wrapping parameters.

                    For a complete list of options, please see [here](
                    https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
                - xla_fsdp_grad_ckpt (`bool`, *optional*, defaults to `False`):
                    Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be
                    used when the xla flag is set to true, and an auto wrapping policy is specified through
                    fsdp_min_num_params or fsdp_transformer_layer_cls_to_wrap.

    """
    pipe: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    pipe_config: Optional[Dict] = field(
        default=None,
        metadata={"help": "Config to be used with pippy (Pytorch Distributed Parallel Training)."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.pipe is None:
            self.pipe = False
        if self.pipe:
            self.parallel_mode = ParallelMode.DISTRIBUTED
        
        if self.pipe_config is None:
            self.pipe_config = {}

        if isinstance(self.pipe_config, str):
            if len(self.pipe) == 0:
                warnings.warn("`--pipe_config` is useful only when `--pipe` is specified.")
            with io.open(self.pipe_config, "r", encoding="utf-8") as f:
                self.pipe_config = json.load(f)
                for k in list(self.pipe_config.keys()):
                    if k.startswith("pipe_"):
                        v = self.pipe_config.pop(k)
                        self.pipe_config[k[5:]] = v
        else:
            self.pipe_config['mode'] = "layers"
            self.pipe_config['layers_name'] = "model.layers"
            self.pipe_config['all_modules'] = ["model.embed_tokens", "model.layers", "model.norm", "lm_head"], # copied from transformers.LlamaForCausalLM
            self.pipe_config['chunks'] = 8



