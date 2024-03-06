import os
import functools
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from datasets import Dataset
from transformers import training_args, logging
from transformers.data.data_collator import DataCollator
from transformers import modeling_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available, 
    is_peft_available,
)

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)

if is_safetensors_available():
    import safetensors.torch

logger = logging.get_logger(__name__)


from . import BaseTrainer, DDPTrainer
from .training_args import TrainingArguments
from .utils import time_test, torch_distributed_zero_first

TRAINING_ARGS_NAME = "training_args.bin"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"


class Trainer(DDPTrainer, BaseTrainer):
    """
    This trainer is designed to be used with PyTorch FSDP (DistributedDataParallel) training, 
    good for development and debugging.
    """
    def __init__(
        self,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.modules.module.Module],
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs,
    ) -> None:
        
        # distributed settings
        self.fsdp = (args.fsdp is not "") and (isinstance(args.fsdp_config, dict) and args.fsdp_config != {})

        # Call the constructor of the base class
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, optimizers, **kwargs)


    def _distributed_setting(self):
        """
        Set up the distributed training environment.
        """
        if self.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend=self.args.ddp_backend if self.args.ddp_backend is not None else "nccl")
                self.args.local_rank = int(os.environ['LOCAL_RANK'])
                self.args.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            master_process = self.rank == 0 # this process will do logging, checkpointing etc.
            if self.fsdp:
                torch.cuda.set_device(torch.device("cuda", self.rank))
        else:
            master_process = True
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        return master_process


    def _wrap_model(self, model):
        """
            Wrap the model to ddp model if the fsdp is enabled
        """
        if self.data_parallel and not isinstance(model, torch.nn.DataParallel):
            model = model.to(self.args.device)
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(self.args.n_gpu)])
        elif self.distributed and not isinstance(model, DDP) and not self.fsdp:
            model = self._create_DDP_model(model)
        elif self.fsdp and not isinstance(model, FSDP):
            model = self._create_FSDP_model(model)
        return model
    
    def _create_FSDP_model(self, model):
        """
            See `transformers.Trainer._wrap_model`
        """
        model.to(torch.device("cuda", self.rank)) if torch.cuda.is_available() else model
        auto_wrap_policy = None
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
            "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
        )

        if self.args.fsdp_config["min_num_params"] > 0:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["min_num_params"]
            )
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception("Could not find the transformer layer class to wrap in the model.")
                else:
                    transformer_cls_to_wrap.add(transformer_cls)

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                # Transformer layer class to wrap
                transformer_layer_cls=transformer_cls_to_wrap,
            )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=self._get_mixed_precision_type(),
            sharding_strategy=self.args.fsdp_config["sharding_strategy"],
            limit_all_gathers=self.args.fsdp_config["limit_all_gathers"],
        )
        return model
    
    def _get_mixed_precision_type(self):
        # requires grad scaler in main loop
        if self.args.fsdp_config["mixed_precision_type"] == "fsdp_fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
        elif self.args.fsdp_config["mixed_precision_type"] == "fsdp_bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )
        elif self.args.fsdp_config["mixed_precision_type"] == "fsdp_bf16_working":
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.args.fsdp_config["mixed_precision_type"] == "fsdp_fp32":
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            raise NotImplementedError("The mixed_precision_type {} is not implemented yet.".format(self.args.fsdp_config["mixed_precision_type"]))


    # The following methods are disable, because they are implemented by FSDP

    def create_autocast(self):
        self.autocast = super().create_autocast()
        if self.fsdp:
            self.autocast = nullcontext()
            self.use_amp = False
        return self.autocast

    def create_scaler(self):
        self.scaler = super().create_scaler()
        if self.fsdp:
            self.scaler = None
            self.use_grad_scaler = False
            if self.args.fsdp_config["mixed_precision_type"] in ["fsdp_fp16"]:
                self.scaler = ShardedGradScaler()
                self.use_grad_scaler = True
        return self.scaler

    # The following methods are related to special saving / loading methods of FSDP

    def _save_checkpoint(self, checkpoint_folder, raw_model, optimizer, lr_scheduler, scaler, train_loss, eval_loss):
        if self.fsdp:
            dist.barrier()
            if self.args.save_only_model:
                self._get_fsdp_state_dict_type(raw_model)
                self._save_model(checkpoint_folder, raw_model)
            else:
                self._save_fsdp_model_optimizer_and_scheduler(checkpoint_folder, raw_model, optimizer, lr_scheduler)
                # Save RNG state
                self._save_rng_state(checkpoint_folder)
                # Save scaler
                if self.use_grad_scaler:
                    self._save_scaler(checkpoint_folder, scaler)
                # Save state
                self._save_state(checkpoint_folder, optimizer, train_loss, eval_loss)
        else:
            # Save model checkpoint
            self._save_model(checkpoint_folder, raw_model)

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(checkpoint_folder, optimizer, lr_scheduler)
                # Save RNG state
                self._save_rng_state(checkpoint_folder)
                # Save scaler
                if self.use_grad_scaler:
                    self._save_scaler(checkpoint_folder, scaler)
                # Save state
                self._save_state(checkpoint_folder, optimizer, train_loss, eval_loss)

    def _save_model(self, output_dir, raw_model, state_dict=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(raw_model, supported_classes):
            if state_dict is None:
                state_dict = raw_model.state_dict()

            if isinstance(raw_model, supported_classes):
                raw_model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            torch.save(raw_model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_fsdp_model_optimizer_and_scheduler(self, checkpoint_folder, raw_model, optimizer, lr_scheduler):
        self._get_fsdp_state_dict_type(raw_model)
        state_dict = raw_model.state_dict()
        self._save_model(checkpoint_folder, raw_model, state_dict)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, OPTIMIZER_NAME))
        torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_folder, SCHEDULER_NAME))


    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        if resume_from_checkpoint is None:
            return
        self._load_model(resume_from_checkpoint, model)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        if self.fsdp:
            self._get_fsdp_state_dict_type(self.model)
            self.model.load_state_dict(self.model.state_dict)
            optim_state_dict = FSDP.optim_state_dict_to_load(
                self.model, self.optimizer, self.optimizer.state_dict
            )
            self.optimizer.load_state_dict(optim_state_dict)
        self._load_rng_state(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)
        self._load_state(resume_from_checkpoint)

    def _get_fsdp_state_dict_type(self, model):
        if self.args.fsdp_config["checkpoint_type"] == StateDictType.FULL_STATE_DICT:
            return FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                # FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
                # FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True),
            )
        elif self.args.fsdp_config["checkpoint_type"] == StateDictType.SHARDED_STATE_DICT:
            return FSDP.set_state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                # ShardedStateDictConfig(offload_to_cpu=False),
                # ShardedOptimStateDictConfig(offload_to_cpu=False),
            )
        else:
            return nullcontext()
