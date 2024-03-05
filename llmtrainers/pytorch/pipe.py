import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.nn.modules.module import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe

from datasets import Dataset
from transformers.data.data_collator import DataCollator
from transformers import modeling_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from . import BaseTrainer, DDPTrainer
from .training_args import TrainingArguments
from .utils import time_test, get_nested_attr, set_nested_attr, has_nested_attr

# try:
#     import pippy
#     from pippy import Pipe, PipeSplitWrapper, annotate_split_points, PipelineStage
# except:
#     raise ImportError(f"The pippy package is required but not installed, please access the repo: https://github.com/pytorch/PiPPy/ to install")


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

        # Call the constructor of the base class
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, optimizers, **kwargs)

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
            if not has_nested_attr(model, module_name):
                raise AttributeError(f"Model does not have attribute {module_name}")
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
