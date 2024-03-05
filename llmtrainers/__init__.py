def trainer_cls(select: str = 'pytorch-ddp'):
    if select == 'pytorch-base':
        from llmtrainers.pytorch import BaseTrainer
        return BaseTrainer
    elif select == 'pytorch-ddp':
        from llmtrainers.pytorch import DDPTrainer
        return DDPTrainer
    elif select == 'pytorch-pipe':
        from llmtrainers.pytorch import PipeTrainer
        return PipeTrainer
    elif select == 'pytorch-fsdp':
        from llmtrainers.pytorch import FSDPTrainer
        return FSDPTrainer
    else:
        raise ValueError(f"Unknown trainer type: {select}")