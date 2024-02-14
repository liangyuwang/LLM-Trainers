import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.distributed as dist
from torch.nn.modules.module import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe

from datasets import Dataset
from transformers.data.data_collator import DataCollator
from transformers import modeling_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from ... import PrivacyEngine, PrivacyConfig
from ...amp import DPGradScaler
from . import BaseDPTrainer, DDPDPTrainer
# from . import training_args, TrainingArguments
from transformers import training_args, TrainingArguments
from .utils import get_nested_attr, set_nested_attr, has_nested_attr

# try:
#     import pippy
#     from pippy import Pipe, PipeSplitWrapper, annotate_split_points, PipelineStage
# except:
#     raise ImportError(f"The pippy package is required but not installed, please access the repo: https://github.com/pytorch/PiPPy/ to install")


class DPTrainer(DDPDPTrainer, BaseDPTrainer):
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
        self.pipeline = args.pipe

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
            distributed=self.pipeline,
        )
        if self.args.fp16 or self.args.bf16:
            if self.args.half_precision_backend == "dp_amp":
                scaler = DPGradScaler()
        return model, optimizer, train_dataloader, scaler


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
        elif self.pipeline:
            if self.args.pipe_config['mode'] == "layers":
                model = self._create_layer_pipeline_from_model(
                    model=model, 
                    layers_name=self.args.pipe_config['layers_name'],
                    all_modules=self.args.pipe_config['all_modules'],
                    splits=self.args.n_gpu, 
                    chunks=self.args.pipe_config['chunks'],
                    rank=int(os.environ['RANK']))
            else:
                raise ValueError(f"Please modify the model to match the torch.distributed.pipeline")
        return model


    def _create_layer_pipeline_from_model(
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
            inner_pipeline_stages = []
            layers = get_nested_attr(model, layers_name)
            if not isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential)):
                raise TypeError(f"Attribute {layers_name} is not a ModuleList or Sequential")
            if len(layers) < splits:
                raise ValueError(f"Layers pipeline requires number of layers is greater than number of splits")

            total_layers = len(layers)
            layers_per_split = total_layers // splits

            for i in range(0, total_layers, layers_per_split):
                sequential_block = torch.nn.Sequential(*layers[i:i + layers_per_split])
                device = torch.device(f'cuda:{i}') if torch.cuda.is_available() else torch.device('cpu')
                sequential_block.to(device)
                inner_pipeline_stages.append(sequential_block)
            
            return inner_pipeline_stages


        pipeline_stages = []
        for module_name in all_modules:
            if not has_nested_attr(model, module_name):
                raise AttributeError(f"Model does not have attribute {module_name}")
            if module_name == layers_name:
                pipeline_stages + layer_pipeline(model, module_name)
            device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
            pipeline_stages.append(get_nested_attr(model, module_name).to(device))

        pipeline_model = Pipe(torch.nn.Sequential(*pipeline_stages), chunks=chunks)
        return pipeline_model

