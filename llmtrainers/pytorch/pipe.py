import os
import inspect
from packaging import version
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed import rpc
from torch.nn.modules.module import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe

import datasets
from datasets import Dataset
from transformers.file_utils import is_datasets_available
from transformers.data.data_collator import DataCollator
from transformers import training_args, modeling_utils, logging
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import RemoveColumnsCollator, seed_worker
from transformers.utils import find_labels

from . import BaseTrainer, DDPTrainer
from .base import _is_peft_model
from .training_args import TrainingArguments
from .utils import time_test, get_nested_attr, set_nested_attr, has_nested_attr

# try:
#     import pippy
#     from pippy import Pipe, PipeSplitWrapper, annotate_split_points, PipelineStage
# except:
#     raise ImportError(f"The pippy package is required but not installed, please access the repo: https://github.com/pytorch/PiPPy/ to install")

logger = logging.get_logger(__name__)


class Trainer(DDPTrainer, BaseTrainer):
    """
    This trainer is designed to be used with PyTorch DDP (DistributedDataParallel) training, 
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
        self.pipeline = args.pipe
        if self.pipeline:
            self.distributed = args.parallel_mode == training_args.ParallelMode.DISTRIBUTED
            self.model = model
            default_label_names = find_labels(self.model.__class__)
            self.label_names = default_label_names if args.label_names is None else args.label_names
            self._signature_columns = None
            self._set_signature_columns_if_needed()
            _signature_columns = self._signature_columns

        # Call the constructor of the base class
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, optimizers, **kwargs)

        self._signature_columns = _signature_columns

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
            # If using ddp, only require backward grad sync on the last micro step
            if self.distributed and not self.pipeline:
                model.require_backward_grad_sync = (micro_step == self.args.gradient_accumulation_steps - 1)
            loss_micro_accumulated += self.training_step(model, inputs, optimizer, scaler, autocast)
        return loss_micro_accumulated / self.args.gradient_accumulation_steps
    
    def _end_training(self):
        if self.distributed:
            dist.destroy_process_group()
        if self.pipeline:
            rpc.shutdown()
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
            rank = int(os.environ['RANK'])
            master_process = rank == 0 # this process will do logging, checkpointing etc.
        else:
            master_process = True
        if self.pipeline:
            rpc.init_rpc(f"worker{int(os.environ['RANK'])}", rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
            # rpc.init_rpc(f"worker", rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        return master_process

    def _wrap_model(self, model):
        """
            Wrap the model to distributed model
        """
        if self.data_parallel and not isinstance(model, torch.nn.DataParallel):
            model = model.to(self.args.device)
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(self.args.n_gpu)])
        elif self.distributed and not isinstance(model, DDP) and not self.pipeline:
            model = self._create_DDP_model(model)
        elif self.pipeline:
            if not isinstance(model, Pipe):
                if self.args.pipe_config['mode'] == "layers":
                    model = self._create_layer_pipeline_from_model(
                        model=model, 
                        layers_name=self.args.pipe_config['layers_name'],
                        all_modules=self.args.pipe_config['all_modules'],
                        splits=self.args.pipe_config['splits'], 
                        chunks=self.args.pipe_config['chunks'],
                        rank=int(os.environ['RANK']))
                else:
                    raise NotImplementedError(f"Pipeline mode {self.args.pipe_config['mode']} is not implemented")
            # self._print_pipeline_model_devices(model)
        return model


    def _create_layer_pipeline_from_model(
        self,
        model, 
        layers_name="model.layers", 
        all_modules=["model.embed_tokens", "model.layers", "model.norm", "lm_head"], # copied from transformers.LlamaForCausalLM
        splits=2, 
        chunks=8,
        rank=0,
    ):
        """
            Layer pipeline is a suitable pipeline strategy for transformers models,
        """
        # Check if the args in all_modules are valid
        model_signature = self._get_signature_columns_if_needed(model)
        for module_name in all_modules:
            if not has_nested_attr(model, module_name):
                raise AttributeError(f"Model does not have attribute {module_name}")
            module = get_nested_attr(model, module_name)
            module_signature = self._get_signature_columns_if_needed(module)
            if module_signature != model_signature:
                raise ValueError(f"Module {module_name} args names {module_signature} does not match the model args names {model_signature}")

        def layer_pipeline(model, layers_name):
            layer_pipeline_stages = []
            layers = get_nested_attr(model, layers_name)
            if not isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential)):
                raise TypeError(f"Attribute {layers_name} is not a ModuleList or Sequential")
            if len(layers) < splits:
                raise ValueError(f"Layers pipeline requires number of layers is greater than number of splits")
            for idx, layer in enumerate(layers):
                cuda_idx = idx % splits
                layer.to(torch.device(f'cuda:{cuda_idx}') if torch.cuda.is_available() else torch.device('cpu'))
                layer_pipeline_stages.append(layer)            
            return layer_pipeline_stages

        pipeline_stages = []
        for module_name in all_modules:
            if module_name == layers_name:
                pipeline_stages + layer_pipeline(model, module_name)
            else:
                device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
                pipeline_stages.append(get_nested_attr(model, module_name).to(device))

        model_blocks = torch.nn.Sequential(*pipeline_stages)
        pipeline_model = Pipe(model_blocks, chunks=chunks)
        return pipeline_model

    def _print_pipeline_model_devices(self, pipeline_model):
        # Check if pipeline_model is an instance of Pipe
        if isinstance(pipeline_model, Pipe):
            # Iterate over each segment
            for segment_idx, segment in enumerate(pipeline_model):
                if self.master_process:
                    print(f"Segment {segment_idx}:")
                # Each segment is a tuple of (stage, batch)
                stage, _ = segment
                # Iterate over modules in the stage
                for module_idx, module in enumerate(stage.modules()):
                    # Avoid printing wrapper modules like Pipe itself
                    if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, Pipe)):
                        continue
                    if self.master_process:
                        print(f"  Module {module_idx}: {module.__class__.__name__} on {next(module.parameters()).device}")
        else:
            if self.master_process:
                print("The provided model is not a Pipe instance.")

    def _get_signature_columns_if_needed(self, module):
        if _is_peft_model(module):
            module = module.get_base_model()
        signature = inspect.signature(module.forward)
        return list(signature.parameters.keys())
    


    def _unwrap_model(self, model):
        """
            Unwrap the model from distributed model
        """
        if (
            (self.distributed or self.data_parallel)
            and not self.pipeline
        ):
            model = model.module
        return model

    def _wrap_data(self, data):
        """
            Wrap the data
        """
        ordered_data = OrderedDict()
        for k, v in data.items():
            ordered_data[k] = v.to(device=self.args.device) if v is not None else None
        return ordered_data
    

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
        outputs = model(*inputs.values())   # key step for pipeline parallelism
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
        
        data_collator = self._set_full_model_columns(data_collator)   # key step for pipeline parallelism

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

        data_collator = self._set_full_model_columns(data_collator)   # key step for pipeline parallelism

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
    
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        # self._set_signature_columns_if_needed()   # key step for pipeline parallelism
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
        # self._set_signature_columns_if_needed()   # key step for pipeline parallelism
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator


    def _set_full_model_columns(self, data_collator: Callable) -> Callable:
        def wrapper(batch):
            batch = data_collator(batch)
            ordered_batch = OrderedDict()
            for key in self._signature_columns:
                ordered_batch[key] = batch.get(key, None)
            return ordered_batch
        return wrapper
