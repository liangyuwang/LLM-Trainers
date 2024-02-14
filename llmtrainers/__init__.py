def trainer_cls(select='pytorch-ddp'):
    if select == 'pytorch-base':
        from llmtrainers.pytorch import BaseTrainer
        return BaseTrainer
    elif select == 'pytorch-ddp':
        from llmtrainers.pytorch import DDPTrainer
        return DDPTrainer
    else:
        raise ValueError(f"Unknown trainer type: {select}")