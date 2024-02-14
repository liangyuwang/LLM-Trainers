import os
import sys
import time
import math
import json
import shutil
import inspect
from packaging import version
import pandas as pd
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
    logging, modeling_utils, training_args, TrainingArguments
)
from transformers.file_utils import is_datasets_available
from transformers import __version__
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer_pt_utils import (
    LabelSmoother,
    LengthGroupedSampler,
    get_dataloader_sampler,
    get_model_param_count,
    get_parameter_names,
)
from transformers.trainer_utils import (
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
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_accelerate_available,
    is_peft_available,
    find_labels,
    logging,
)

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)

# Name of the files used for checkpointing
TRAINER_STATE_NAME = transformers.trainer.TRAINER_STATE_NAME

from .utils import find_executable_batch_size

logger = logging.get_logger(__name__)


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
        self.last_epoch = 0
        self.last_step = 0
        self.best_val_loss = float('inf')

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
    
        # Other settings ...
        self._signature_columns = None
        if self.master_process:
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)


    def train(self, resume_from_checkpoint: Optional[str] = None):
        # Set up the optimizer, lr_scheduler, and dataloader
        optimizer = self.create_optimizer()
        lr_scheduler = self.create_scheduler(num_training_steps=self.args.max_steps, optimizer=optimizer)
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_eval_dataloader()
        # resume from checkpoint
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
        model = self.model

        # Set up amp gradient scaling
        autocast = nullcontext()
        scaler = None
        if self.args.fp16 or self.args.bf16:
            autocast = torch.cuda.amp.autocast(dtype=torch.float16 if self.args.fp16 else torch.bfloat16)
            scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True

        # Compile the model with torch.jit
        if self.args.torch_compile:
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                if self.master_process:
                    print("Compiling the model with torch.jit... (takes a ~minute)")
                model = torch.compile(model)

        # Train the model with automatic batch size finding
        # inner_training_loop = find_executable_batch_size(
        #     self._inner_training_loop, self._train_batch_size, self.args.auto_find_batch_size
        # )
        self._inner_training_loop(train_dataloader, val_dataloader, model, optimizer, lr_scheduler, scaler, autocast)

        # Clean up the training environment
        self._end_training()
    

    def _inner_training_loop(self, train_dataloader, val_dataloader, model, optimizer, lr_scheduler, scaler, autocast):
        
        max_steps, num_train_epochs, num_update_steps_per_epoch = self._training_args(train_dataloader)

        global_step = self.last_step
        best_val_loss = self.best_val_loss
        val_loss = 0.0
        tqdm_bar = nullcontext()
        if not self.args.disable_tqdm:
            tqdm_bar = tqdm(total=max_steps, initial=self.last_step, desc="Training")
        with tqdm_bar as pbar:
            for epoch in range(num_train_epochs - int(self.last_epoch)):
                if global_step > max_steps:
                    break
                train_loss_per_epoch = 0.0
                train_dataloader_iter = iter(train_dataloader)
                for step in range(num_update_steps_per_epoch):
                    if global_step > max_steps:
                        break

                    model.train()

                    # Perform a training micro step
                    train_step_loss = self.training_micro_step(train_dataloader_iter, model, optimizer, scaler, autocast)

                    # Update the training state
                    global_epoch, global_step, train_loss_per_epoch = \
                        self._training_update(model, optimizer, scaler, lr_scheduler, \
                                              pbar, epoch, global_step, step, num_update_steps_per_epoch, \
                                              train_loss_per_epoch, train_step_loss)

                    # evaluate the model
                    val_loss, best_val_loss = self._evaluate_log(val_dataloader, model, optimizer, \
                                                                 global_epoch, global_step, \
                                                                 train_step_loss, val_loss, best_val_loss)
                    
                    # save the training state with steps / per epoch
                    self._save(self.args.output_dir, model, optimizer, lr_scheduler, \
                               global_epoch, global_step, \
                               train_step_loss, val_loss, best_val_loss)


    def training_micro_step(self, train_dataloader, model, optimizer, scaler, autocast):
        """
        Perform a single training micro step.
        """
        loss_micro_accumulated = 0.0
        for micro_step in range(self.args.gradient_accumulation_steps):
            try:
                inputs = next(train_dataloader) # When you use next(train_dataloader) within the training_micro_step function, the state of train_dataloader in the _inner_training_loop will also change
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
            if self.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss = loss.item()
            return step_loss


    def _training_update(self, model, optimizer, scaler, lr_scheduler, pbar, epoch, global_step, step, num_update_steps_per_epoch, train_loss_per_epoch, train_step_loss):
        """
        Perform the update step for the training loop.
        """
        # Gradient clipping
        if self.args.max_grad_norm != 0.0:
            if self.use_amp:
                scaler.unscale_(optimizer)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        
        # update the model parameters
        if self.use_amp:
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
        global_epoch = int(self.last_epoch) + epoch + step / num_update_steps_per_epoch
        global_step += 1
        
        # update the training loss
        train_loss_per_epoch += train_step_loss / num_update_steps_per_epoch

        return global_epoch, global_step, train_loss_per_epoch
    

    def _training_args(self, train_dataloader):
        """
            See `transformers.Trainer._inner_training_loop`
        """
        # Compute epoch, steps, and samples info for the training loop
        total_train_batch_size = self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            # num_examples = self.num_examples(train_dataloader)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
                # # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # # the best we can do.
                # num_train_samples = self.args.max_steps * total_train_batch_size
                # if self.args.include_tokens_per_second:
                #     num_train_tokens = (
                #         self.num_tokens(train_dataloader, self.args.max_steps) * self.args.gradient_accumulation_steps
                #     )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
                # num_train_samples = self.num_examples(train_dataloader) * self.args.num_train_epochs
                # if self.args.include_tokens_per_second:
                #     num_train_tokens = self.num_tokens(train_dataloader) * self.args.num_train_epochs
        elif self.args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = self.args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            # num_examples = total_train_batch_size * self.args.max_steps
            # num_train_samples = self.args.max_steps * total_train_batch_size
            # if self.args.include_tokens_per_second:
            #     num_train_tokens = self.num_tokens(train_dataloader, self.args.max_steps) * self.args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {self.args.max_steps}"
            )
        return max_steps, num_train_epochs, num_update_steps_per_epoch

    def _end_training(self):
        if self.master_process:
            print("Training completed.")


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
    def _evaluate_log(self, val_dataloader, model, optimizer, global_epoch, global_step, train_step_loss, val_loss, best_val_loss):
        """
            Evaluate the model and log the evaluation results. Used inside the training loop.
        """
        model.eval()
        if self.args.evaluation_strategy == 'steps':
            if self.args.eval_steps > 0 and (global_step + 1) % self.args.eval_steps == 0:
                val_loss = self._inner_evaluate_loop(val_dataloader, model)
                if self.master_process:
                    # update the training info
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    if self.args.disable_tqdm:
                        print(f"step {global_step+1}: train loss {train_step_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}, lr {optimizer.param_groups[0]['lr']}, epoch {global_epoch: .2f}")
                    else:
                        tqdm.write(f"step {global_step+1}: train loss {train_step_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}, lr {optimizer.param_groups[0]['lr']}, epoch {global_epoch: .2f}")
        elif self.args.evaluation_strategy == 'no':
            if self.master_process:
                if self.args.logging_steps > 0 and (global_step + 1) % self.args.logging_steps == 0:
                    if self.args.disable_tqdm:
                        print(f"step {global_step+1}: train loss {train_step_loss:.4f}, lr {optimizer.param_groups[0]['lr']}, epoch {global_epoch: .2f}")
                    else:
                        tqdm.write(f"step {global_step+1}: train loss {train_step_loss:.4f}, lr {optimizer.param_groups[0]['lr']}, epoch {global_epoch: .2f}")
        else:
            raise NotImplementedError("Evaluation strategy must be 'no', 'steps', or 'epoch'.")
        return val_loss, best_val_loss

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
                eval_loss += self.compute_loss(model, inputs).mean()
                if not self.args.disable_tqdm:
                    pbar.update(1)
        return eval_loss / len(eval_dataloader)


    def _save(self, out_dir, model, optimizer, lr_scheduler, global_epoch, global_step, train_loss, val_loss, best_val_loss):
        save = self.args.save_strategy == "steps" and (global_step + 1) % self.args.save_steps == 0
        save = save or (self.args.save_strategy == "epoch" and global_epoch-1==0)
        
        if save and self.master_process:
            self._save_logs(out_dir, optimizer, lr_scheduler, global_epoch, global_step, train_loss, val_loss, best_val_loss)
            
            out_dir = os.path.join(out_dir, f'checkpoint-{global_step+1}')
            if self.master_process:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            raw_model = self._unwrap_model(model)   # unwrap the model from distributed model
            self._save_checkpoint(
                out_dir, raw_model, optimizer, lr_scheduler, global_epoch, global_step, train_loss, val_loss, best_val_loss
            )
            
            if self.args.save_total_limit is not None:
                all_elements = os.listdir(self.args.output_dir)
                step_checkpoints = [ckpt for ckpt in all_elements if ckpt.startswith("step-")]
                if len(step_checkpoints) > self.args.save_total_limit:
                    step_checkpoints = sorted(step_checkpoints, key=lambda x: int(x.split("-")[1]))
                    shutil.rmtree(os.path.join(self.args.output_dir, step_checkpoints[0]))


    def _save_checkpoint(self, out_dir, raw_model, optimizer, lr_scheduler, global_epoch, global_step, train_loss, val_loss, best_val_loss):
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': global_epoch,
            'step': global_step+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'is_the_best': val_loss == best_val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    def _save_logs(self, out_dir, optimizer, lr_scheduler, global_epoch, global_step, train_loss, val_loss, best_val_loss):
        logs_file = os.path.join(out_dir, 'train_logs.json')
        if os.path.exists(logs_file):
            with open(logs_file, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []  # If the file is empty or corrupted, start with an empty list
        else:
            logs = []
        new_log = {
            'epoch': global_epoch,
            'step': global_step+1,
            'train_loss': train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss,
            'val_loss': val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
            'best_val_loss': best_val_loss.item() if isinstance(best_val_loss, torch.Tensor) else best_val_loss,
            'lr': optimizer.param_groups[0]['lr'],
        }
        logs.append(new_log)
        with open(os.path.join(out_dir, 'train_logs.json'), 'w') as f:
            json.dump(logs, f, indent=4)

    def _load_from_checkpoint(self, checkpoint_path: str):
        print(f"loaded checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model = self._unwrap_model(self.model)
        model.load_state_dict(checkpoint['model'])
        self.model = self._wrap_model(model)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.last_epoch = checkpoint['epoch']
        self.last_step = checkpoint['step']-1
        self.best_val_loss = checkpoint['best_val_loss']
        checkpoint = None # free up memory


    def _distributed_setting(self):
        """
        Set up the distributed training environment.
        """
        master_process = True
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        return master_process

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


    # The following methods are copied from Huggingface's Trainer class

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
    

    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            if self.is_fsdp_enabled:
                load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, model, resume_from_checkpoint)
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        weights_only=is_torch_greater_or_equal_than_1_13,
                    )

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )


    def _issue_warnings_after_load(self, load_result):
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
