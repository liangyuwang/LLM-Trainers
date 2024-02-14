import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import Dataset
from transformers.data.data_collator import DataCollator
from transformers import modeling_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from ... import PrivacyEngine, PrivacyConfig
from ...amp import DPGradScaler
from . import BaseDPTrainer
# from . import training_args, TrainingArguments
from transformers import training_args, TrainingArguments


class DPTrainer(BaseDPTrainer):
    """
    This trainer is designed to be used with PyTorch DDP (DistributedDataParallel) training, 
    good for development and debugging.
    """
    def __init__(
        self,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.modules.module.Module],
        args: TrainingArguments,
        privacy_args: PrivacyConfig,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs,
    ) -> None:
        
        # distributed settings
        self.distributed = args.parallel_mode == training_args.ParallelMode.DISTRIBUTED

        # Call the constructor of the base class
        super().__init__(model, args, privacy_args, data_collator, train_dataset, eval_dataset, tokenizer, optimizers, **kwargs)
    

    def make_private(self, model, optimizer, data_loader, scaler):
        """
        Make the model, optimizer, and data loader private for differential privacy training.
        """
        self.privacy_args.poisson_sampling = False
        privacy_engine = PrivacyEngine(config=self.privacy_args)
        model, optimizer, train_dataloader = privacy_engine.make_private(
            model=model, 
            optimizer=optimizer, 
            data_loader=data_loader,
            distributed=self.distributed,
        )
        if self.args.fp16 or self.args.bf16:
            if self.args.half_precision_backend == "dp_amp":
                scaler = DPGradScaler()
        return model, optimizer, train_dataloader, scaler

    def training_micro_step(self, train_dataloader, model, optimizer, scaler, autocast):
        """
        Perform a single training step.
        """
        loss_micro_accumulated = 0.0
        for micro_step in range(self.args.gradient_accumulation_steps):
            try:
                inputs = next(train_dataloader)
            except StopIteration:
                break
            inputs = self._wrap_data(inputs)
            # If using ddp, only require backward grad sync on the last micro step
            if self.distributed:
                model.require_backward_grad_sync = (micro_step == self.args.gradient_accumulation_steps - 1)
            loss_micro_accumulated += self.training_step(model, inputs, optimizer, scaler, autocast)
        return loss_micro_accumulated / self.args.gradient_accumulation_steps
    
    def _end_training(self):
        if self.distributed:
            dist.destroy_process_group()
        if self.master_process:
            print("Training completed.")
    
    def _distributed_setting(self):
        """
        Set up the distributed training environment.
        """
        if self.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend=self.args.ddp_backend if self.args.ddp_backend is not None else "nccl")
                self.args.local_rank = int(os.environ['LOCAL_RANK'])
                self.args.world_size = int(os.environ['WORLD_SIZE'])
            ddp_rank = int(os.environ['RANK'])
            master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        else:
            master_process = True
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        return master_process

    def _wrap_model(self, model):
        """
            Wrap the model to ddp model if the ddp is enabled
        """
        if self.data_parallel and not isinstance(model, torch.nn.DataParallel):
            model = model.to(self.args.device)
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(self.args.n_gpu)])
        elif self.distributed and not isinstance(model, DDP):
            model = self._create_DDP_model(model)
        return model
    
    def _unwrap_model(self, model):
        """
            Unwrap the model from ddp model if the ddp is enabled
        """
        model = model.module if self.distributed or self.data_parallel else model
        return model


    def _create_DDP_model(self, model):
        model = model.to(f"cuda:{self.args.local_rank % torch.cuda.device_count()}")
        # copied from transformers.Trainer
        kwargs = {}
        if self.args.ddp_find_unused_parameters is not None:
            kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
        elif isinstance(model, PreTrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
        else:
            kwargs["find_unused_parameters"] = True
        if self.args.ddp_bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
        if self.args.ddp_broadcast_buffers is not None:
            kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers
        model = DDP(model, device_ids=[self.args.local_rank], **kwargs)
        return model
