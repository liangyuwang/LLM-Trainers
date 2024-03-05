import os
import sys
import time
import math
import json
import shutil
import inspect
from packaging import version
import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import datasets
from datasets import Dataset
import transformers
from transformers import (
    logging, modeling_utils, training_args
)
from transformers.file_utils import is_datasets_available
from transformers import __version__
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer import TrainerState
from transformers.trainer_pt_utils import (
    LabelSmoother,
    LengthGroupedSampler,
    IterableDatasetShard,
    get_dataloader_sampler,
    get_model_param_count,
    get_parameter_names,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    HPSearchBackend,
    TrainOutput,
    EvalPrediction,
    RemoveColumnsCollator,
    has_length,
    speed_metrics,
    find_executable_batch_size,
    seed_worker,
)
from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_accelerate_available,
    is_peft_available,
    is_safetensors_available,
    find_labels,
    logging,
)

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)

# Name of the files used for checkpointing
TRAINER_STATE_NAME = transformers.trainer.TRAINER_STATE_NAME

from .utils import time_test, find_executable_batch_size, torch_distributed_zero_first
from .training_args import TrainingArguments

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class Trainer:
    """
        This is a lightweight version of the Trainer class compared with Huggingface's transformers Trainer, 
        and it receive the same arguments as the Huggingface's Trainer.
        This trainer is designed to be used with naive PyTorch training (CPU, GPU, or DataPallel), good for development and debugging.
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
        
        # Init args
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizers = optimizers

        # distributed settings
        self.data_parallel = self.args.parallel_mode == training_args.ParallelMode.NOT_DISTRIBUTED
        self.master_process = self._distributed_setting()
        self.model = self._wrap_model(self.model)   # wrap the model
        
        # training args
        self.optimizer = None
        self.lr_scheduler = None
        self._train_batch_size = self.args.train_batch_size
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.use_amp = False
        self.use_grad_scaler = False
        self.state = TrainerState()
        self.state.epoch = 0
        self.state.global_step = 0
        self.state.best_metric = float('inf')
        self.state.train_batch_size = self._train_batch_size
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
    
        # Other settings ...
        self._signature_columns = None
        self.last_checkpoint_folder = None
        if self.master_process:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)


    # The following methods are related to training

    def train(self, resume_from_checkpoint: Optional[str] = None):
        # Set up the optimizer, lr_scheduler, and dataloader
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(num_training_steps=self.args.max_steps, optimizer=self.optimizer)
        self.autocast = self.create_autocast()
        self.scaler = self.create_scaler()
        # resume from checkpoint
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            if self.state.train_batch_size is not None:
                self._train_batch_size = self.state.train_batch_size
            self.state.log_history = None # free up memory
        
        # Compile the model with torch.jit
        if self.args.torch_compile:
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                if self.master_process:
                    print("Compiling the model with torch.jit... (takes a ~minute)")
                self.model = torch.compile(self.model)

        # Train the model with automatic batch size finding
        # inner_training_loop = find_executable_batch_size(
        #     self._inner_training_loop, self._train_batch_size, self.args.auto_find_batch_size
        # )
        self._inner_training_loop()

        # Clean up the training environment
        self._end_training()
    

    def _inner_training_loop(self):
        model, optimizer, lr_scheduler, scaler, autocast = self.model, self.optimizer, self.lr_scheduler, self.scaler, self.autocast
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        max_steps, num_train_epochs, num_update_steps_per_epoch, num_train_samples, num_train_tokens = self._training_args(train_dataloader)
        last_global_epoch = self.state.epoch
        last_global_step = self.state.global_step
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        # for inputs in iter(self.train_dataset):
        #     print("check inputs: ", inputs.keys(), len(inputs['input_ids']))
        # print("check train_dataloader 1: ", type(train_dataloader), len(train_dataloader))
        # print("check train_dataloader 2: ", train_dataloader.dataset[0])
        # exit()

        eval_loss = 0.0
        tqdm_bar = nullcontext()
        if not self.args.disable_tqdm:
            tqdm_bar = tqdm(total=max_steps, initial=last_global_step, desc="Training")
        if not self.args.overwrite_output_dir:
            if os.listdir(self.args.output_dir):
                exit(f"The save path {self.args.output_dir} is not empty, please clear the folder")
        with tqdm_bar as pbar:
            for epoch in range(num_train_epochs - int(last_global_epoch)):
                if self.state.global_step > max_steps:
                    break
                train_loss_per_epoch = 0.0
                train_dataloader_iter = iter(train_dataloader)
                for step in range(num_update_steps_per_epoch):
                    if self.state.global_step > max_steps:
                        break

                    model.train()

                    # Perform a training micro step
                    train_step_loss, training_time = \
                        self.training_micro_step(train_dataloader_iter, model, optimizer, scaler, autocast)

                    # Update the training state
                    train_loss_per_epoch, updating_time = \
                        self._training_update(model, optimizer, scaler, lr_scheduler, \
                                              pbar, epoch, step, last_global_epoch, num_update_steps_per_epoch, \
                                              train_loss_per_epoch, train_step_loss)

                    # evaluate the model
                    eval_loss = self._evaluate_log(eval_dataloader, model, optimizer, 
                                                   train_step_loss, eval_loss, 
                                                   training_time+updating_time,
                                                   max_steps, num_train_epochs, num_train_samples, num_train_tokens)
                    
                    # save the training state with steps / per epoch
                    self._save(self.args.output_dir, model, optimizer, lr_scheduler, scaler, \
                               train_step_loss, eval_loss)

    @time_test
    def training_micro_step(self, train_dataloader, model, optimizer, scaler, autocast):
        """
        Perform a single training micro step.
        """
        loss_micro_accumulated = 0.0
        for micro_step in range(self.args.gradient_accumulation_steps):
            try:
                inputs = next(train_dataloader)
            except StopIteration:
                break
            inputs = self._wrap_data(inputs)
            loss_micro_accumulated += self.training_step(model, inputs, optimizer, scaler, autocast)
        return loss_micro_accumulated / self.args.gradient_accumulation_steps


    def training_step(self, model, inputs, optimizer, scaler, autocast):
        """
        Perform a single training step.
        """
        with autocast:
            loss = self.compute_loss(model, inputs).mean()
            if self.use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss = loss.item()
            return step_loss


    @time_test
    def _training_update(self, model, optimizer, scaler, lr_scheduler, pbar, inner_epoch, inner_step, last_global_epoch, num_update_steps_per_epoch, train_loss_per_epoch, train_step_loss):
        """
        Perform the update step for the training loop.
        """
        # Gradient clipping
        if self.args.max_grad_norm != 0.0:
            if self.use_grad_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        
        # update the model parameters
        if self.use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # update tqdm bar
        if not self.args.disable_tqdm:
            pbar.update(1)
        
        # Update the epoch, step
        self.state.epoch = int(last_global_epoch) + inner_epoch + inner_step / num_update_steps_per_epoch
        self.state.global_step += 1
        
        # update the training loss
        train_loss_per_epoch += train_step_loss / num_update_steps_per_epoch

        return train_loss_per_epoch


    def _training_args(self, train_dataloader):
        """
            See `transformers.Trainer._inner_training_loop`
        """
        # Compute epoch, steps, and samples info for the training loop
        total_train_batch_size = self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = self.args.max_steps * total_train_batch_size
                num_train_tokens = (
                    self.num_tokens(train_dataloader, self.args.max_steps) * self.args.gradient_accumulation_steps
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * self.args.num_train_epochs
                num_train_tokens = self.num_tokens(train_dataloader) * self.args.num_train_epochs
        elif self.args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = self.args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_samples = self.args.max_steps * total_train_batch_size
            num_train_tokens = self.num_tokens(train_dataloader, self.args.max_steps) * self.args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {self.args.max_steps}"
            )
        return max_steps, num_train_epochs, num_update_steps_per_epoch, num_train_samples, num_train_tokens

    def _end_training(self):
        if self.master_process:
            print("Training completed.")


    # The following methods are related to evaluation

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """
            Evaluate the model on the evaluation dataset.
        """
        if eval_dataset is not None:
            self.eval_dataset = eval_dataset
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        eval_loss = self._inner_evaluate_loop(eval_dataloader, self.model)
        result = {"eval_loss": eval_loss}
        return result

    @torch.no_grad()
    def _evaluate_log(self, eval_dataloader, model, optimizer, train_step_loss, eval_loss, training_time, max_steps, num_train_epochs, num_train_samples, num_train_tokens):
        """
            Evaluate the model and log the evaluation results. Used inside the training loop.
        """
        model.eval()
        should_eval_steps = (self.args.evaluation_strategy == 'steps') and (self.args.eval_steps > 0) and (self.state.global_step + 1) % self.args.eval_steps == 0
        should_eval_epoch = (self.args.evaluation_strategy == 'epoch') and (int(self.state.epoch) == self.state.epoch)
        train_tokens_per_step = num_train_tokens / max_steps
        train_samples_per_step = num_train_samples / max_steps
        if should_eval_steps or should_eval_epoch:
            eval_loss = self._inner_evaluate_loop(eval_dataloader, model)
            self.state.best_metric = eval_loss if eval_loss < self.state.best_metric else self.state.best_metric
            if self.master_process:
                if self.args.disable_tqdm:
                    print(f"step {self.state.global_step+1}: train-loss: {train_step_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}, epoch: {self.state.epoch: .2f}")
                    print(f"  eval-loss: {eval_loss:.4f}, best-eval-loss: {self.state.best_metric:.4f}")
                    print(f"  training time per step: {train_tokens_per_step/training_time:.4f} tokens/s, {train_samples_per_step/training_time:.4f} samples/s")
                else:
                    tqdm.write(f"step {self.state.global_step+1}: train-loss: {train_step_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}, epoch: {self.state.epoch: .2f}")
                    tqdm.write(f"  eval-loss: {eval_loss:.4f}, best-eval-loss: {self.state.best_metric:.4f}")
                    tqdm.write(f"  training time per step: {train_tokens_per_step/training_time:.4f} tokens/s, {train_samples_per_step/training_time:.4f} samples/s")
        elif self.args.evaluation_strategy == 'no':
            if self.master_process:
                if self.args.logging_steps > 0 and (self.state.global_step + 1) % self.args.logging_steps == 0:
                    if self.args.disable_tqdm:
                        print(f"step {self.state.global_step+1}: train-loss: {train_step_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}, epoch: {self.state.epoch: .2f}")
                        print(f"  training time per step: {train_tokens_per_step/training_time:.4f} tokens/s, {train_samples_per_step/training_time:.4f} samples/s")
                    else:
                        tqdm.write(f"step {self.state.global_step+1}: train-loss: {train_step_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}, epoch: {self.state.epoch: .2f}")
                        tqdm.write(f"  training time per step: {train_tokens_per_step/training_time:.4f} tokens/s, {train_samples_per_step/training_time:.4f} samples/s")
        else:
            pass
        return eval_loss

    @torch.no_grad()
    def _inner_evaluate_loop(self, eval_dataloader, model):
        """
            Evaluate the model on the evaluation dataset. Used inside the training loop.
        """
        tqdm_bar = nullcontext()
        if not self.args.disable_tqdm:
            tqdm_bar = tqdm(total=len(eval_dataloader), desc="Evaluating", leave=False)
        eval_loss = 0.0
        with tqdm_bar as pbar:
            for _, inputs in enumerate(eval_dataloader):
                inputs = self._wrap_data(inputs)
                inner_eval_loss = self.compute_loss(model, inputs).mean()
                eval_loss = self._evaluate_update(pbar, eval_loss, inner_eval_loss.item())
        return eval_loss / len(eval_dataloader)
    
    @torch.no_grad()
    def _evaluate_update(self, pbar, eval_loss, inner_eval_loss):
        """
            Update the evaluation loop. Used inside the training loop.
        """
        if not self.args.disable_tqdm:
            pbar.update(1)
        self.state.eval_steps += 1
        return eval_loss + inner_eval_loss


    # The following methods are related to distributed / single device settings

    def _distributed_setting(self):
        """
        Set up the distributed training environment.
        """
        master_process = True
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        return master_process
    
    def _devices_sync(self):
        """
        Synchronize the devices.
        """
        pass

    def _rank_zero_first(self):
        """
        Execute the code in the 0 rank first.
        Example:
        >>> with _rank_zero_first():
        >>>    ... # the code that needs to be executed in the 0 rank first
        """
        return nullcontext()

    def _wrap_model(self, model):
        """
            Wrap the model
        """
        if self.data_parallel and not isinstance(model, torch.nn.DataParallel):
            model = model.to(self.args.device)
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(self.args.n_gpu)])
        return model
    
    def _unwrap_model(self, model):
        """
            Unwrap the model
        """
        return model.module if self.data_parallel else model
    
    def _wrap_data(self, data):
        """
            Wrap the data
        """
        data = data.to(self.args.device)
        return data
    
    def _unwrap_data(self, data):
        raise NotImplementedError


    # The following methods are copied from Huggingface's Trainer class, related to training and evaluation

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        See `transformers.Trainer.compute_loss`:
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        You can also override this function in a subclass to support custom model input (for example, no 'input_id').
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if _is_peft_model(model):
                model_name = model.base_model.model._get_name()
            else:
                model_name = model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                        lr=self.args.learning_rate, 
                                        betas=(self.args.adam_beta1, self.args.adam_beta2), 
                                        weight_decay=self.args.weight_decay)
        return self.optimizer
    

    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    
    # The following methods are created, related to mixed precision

    def create_autocast(self):
        """
        Create the autocast context manager. This is a no-op if the trainer is not using mixed precision.
        """
        if self.args.fp16 or self.args.bf16:
            if not torch.cuda.is_available():
                raise ValueError("Amp can only be used with CUDA.")
            self.autocast = torch.cuda.amp.autocast(dtype=torch.float16 if self.args.fp16 else torch.bfloat16)
            self.use_amp = True
        else:
            self.autocast = nullcontext()
        return self.autocast

    def create_scaler(self):
        """
        Create the autocast context manager. This is a no-op if the trainer is not using mixed precision.
        """
        if self.args.fp16:
            if not torch.cuda.is_available():
                raise ValueError("Amp can only be used with CUDA.")
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_grad_scaler = True
        else:
            self.scaler = None
        return self.scaler

    # The following methods are copied from Huggingface's Trainer class, related to data loading

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)


    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(eval_dataset, **dataloader_params)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self._unwrap_model(self.model)
            if _is_peft_model(self.model):
                model_to_inspect = self.model.get_base_model()
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
    
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)
    
    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        # Deprecated code
        if self.args.use_legacy_prediction_loop or self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None
    
    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens


    # The following methods are copied and adjusted from Huggingface's Trainer class, related to saving and loading
    
    def _save(self, out_dir, model, optimizer, lr_scheduler, scaler, train_loss, eval_loss):
        save = self.args.save_strategy == "steps" and (self.state.global_step + 1) % self.args.save_steps == 0
        save = save or (self.args.save_strategy == "epoch" and self.state.epoch-1==0)
        if self.args.save_strategy == "no":
            save = False
        
        with self._rank_zero_first():   # execute the code in the zero process first
            if save and self.master_process and self.args.should_save:
            # if save and self.master_process:
                checkpoint_folder = os.path.join(out_dir, f'checkpoint-{self.state.global_step+1}')
                if self.master_process:
                    if not os.path.exists(checkpoint_folder):
                        os.makedirs(checkpoint_folder)

                raw_model = self._unwrap_model(model)   # unwrap the model from distributed model
                self._save_checkpoint(
                    checkpoint_folder, raw_model, optimizer, lr_scheduler, scaler, train_loss, eval_loss
                )

                best_eval_loss = self.state.best_metric
                if eval_loss == best_eval_loss:
                    self.state.best_model_checkpoint = checkpoint_folder
                    best_model_checkpoint = os.path.join(out_dir, 'best_model')
                    self._save_checkpoint(
                        best_model_checkpoint, raw_model, optimizer, lr_scheduler, scaler, train_loss, eval_loss
                    )
                
                if self.args.save_total_limit is not None:
                    all_elements = os.listdir(self.args.output_dir)
                    step_checkpoints = [ckpt for ckpt in all_elements if ckpt.startswith("checkpoint-")]
                    if len(step_checkpoints) > self.args.save_total_limit:
                        step_checkpoints = sorted(step_checkpoints, key=lambda x: int(x.split("-")[1]))
                        shutil.rmtree(os.path.join(self.args.output_dir, step_checkpoints[0]))
                
                self.last_checkpoint_folder = checkpoint_folder
    
    def _save_checkpoint(self, checkpoint_folder, raw_model, optimizer, lr_scheduler, scaler, train_loss, eval_loss):
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
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            raw_model.save_pretrained(
                output_dir, is_main_process=self.args.should_save, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_optimizer_and_scheduler(self, output_dir, optimizer, lr_scheduler):
        torch.save(optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
    
    def _save_rng_state(self, output_dir):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

    def _save_scaler(self, output_dir, scaler):
        torch.save(scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

    def _save_state(self, output_dir, optimizer, train_loss, eval_loss):
        # Save metrics
        train_metrics = {
            "epoch": self.state.epoch,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "loss": train_loss,
            "step": self.state.global_step
        }
        eval_metrics = {
            "epoch": self.state.epoch,
            "eval_loss": eval_loss,
            "step": self.state.global_step
        }
        if self.last_checkpoint_folder == None:
            self.state.log_history = [train_metrics, eval_metrics]
        else:
            state = TrainerState.load_from_json(os.path.join(self.last_checkpoint_folder, TRAINER_STATE_NAME))
            self.state.log_history = state.log_history
            self.state.log_history.append(train_metrics)
            self.state.log_history.append(eval_metrics)
        path = os.path.join(output_dir, TRAINER_STATE_NAME)
        self.state.save_to_json(path)
        self.state.log_history = None # free up memory


    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        if resume_from_checkpoint is None:
            return
        self._load_model(resume_from_checkpoint, model)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_rng_state(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)
        self._load_state(resume_from_checkpoint)

    def _load_model(self, checkpoint, model=None):
        if model is None:
            model = self._unwrap_model(self.model)

        config_file = os.path.join(checkpoint, CONFIG_NAME)
        weights_file = os.path.join(checkpoint, WEIGHTS_NAME)
        safe_weights_file = os.path.join(checkpoint, SAFE_WEIGHTS_NAME)

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                ]
            )
        ):
            raise ValueError(f"Can't find a valid checkpoint at {checkpoint}")

        logger.info(f"Loading model from {checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        # We load the model state dict on the CPU to avoid an OOM error.
        if self.args.save_safetensors and os.path.isfile(safe_weights_file):
            state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
        else:
            state_dict = torch.load(
                weights_file,
                map_location="cpu",
                weights_only=is_torch_greater_or_equal_than_1_13,
            )
        model.load_state_dict(state_dict)
        self.model = self._wrap_model(model)
        del state_dict  # free memory

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return
        optimizer_file = os.path.join(checkpoint, OPTIMIZER_NAME)
        scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME)
        if not any(
            os.path.isfile(f)
            for f in [
                optimizer_file,
                scheduler_file,
            ]
        ):
            logger.info("No optimizer and scheduler state found. Only loading model.")
            return
        if os.path.isfile(optimizer_file):
            self.optimizer.load_state_dict(torch.load(optimizer_file))
        if os.path.isfile(scheduler_file):
            self.lr_scheduler.load_state_dict(torch.load(scheduler_file))

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def _load_scaler(self, checkpoint):
        if not self.use_grad_scaler:
            return
        if checkpoint is None:
            return
        if not os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
            logger.info("No scaler state found. Only loading model.")
            return
        self.scaler.load_state_dict(torch.load(os.path.join(checkpoint, SCALER_NAME)))

    def _load_state(self, checkpoint):
        if checkpoint is None:
            return
        if not os.path.isfile(os.path.join(checkpoint, TRAINER_STATE_NAME)):
            logger.info("No trainer state found. Only loading model.")
            return
        self.state = TrainerState.load_from_json(os.path.join(checkpoint, TRAINER_STATE_NAME))


    # The following methods are copied from Huggingface's Trainer class, related to distributed training

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.args.process_index == 0
