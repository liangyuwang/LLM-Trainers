import os
import functools
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
from transformers import training_args
from transformers.data.data_collator import DataCollator
from transformers import modeling_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name

from . import BaseTrainer, DDPTrainer
from .training_args import TrainingArguments
from .utils import torch_distributed_zero_first

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
        if self.fsdp:
            torch.cuda.set_device(torch.device("cuda", args.local_rank))

        # Call the constructor of the base class
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, optimizers, **kwargs)


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
        model.to(torch.device("cuda"))
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
            device_id=torch.cuda.current_device(),
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
        return self.scaler

    # The following methods are related to special saving / loading methods of FSDP

    def _save_checkpoint(self, checkpoint_folder, raw_model, optimizer, lr_scheduler, scaler, train_loss, eval_loss):
        if self.fsdp:
            if self.args.save_only_model:
                FSDP.set_state_dict_type(
                    raw_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
                    FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True),
                )
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

    def _save_fsdp_model_optimizer_and_scheduler(self, checkpoint_folder, raw_model, optimizer, lr_scheduler):
        FSDP.set_state_dict_type(
            raw_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True),
        )
        state_dict = raw_model.state_dict()
        self._save_model(checkpoint_folder, raw_model, state_dict)
        original_osd = optimizer.state_dict()
        optim_state_dict = FSDP.optim_state_dict(
            raw_model,
            optimizer,
            optim_state_dict=original_osd
        )
        torch.save(optim_state_dict , os.path.join(checkpoint_folder, OPTIMIZER_NAME))
        torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_folder, SCHEDULER_NAME))


    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        if resume_from_checkpoint is None:
            return
        self._load_model(resume_from_checkpoint, model)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        if self.fsdp:
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True),
            )
            self.model.load_state_dict(self.model.state_dict)
            optim_state_dict = FSDP.optim_state_dict_to_load(
                self.model, self.optimizer, self.optimizer.state_dict
            )
            self.optimizer.load_state_dict(optim_state_dict)
        self._load_rng_state(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)
        self._load_state(resume_from_checkpoint)
