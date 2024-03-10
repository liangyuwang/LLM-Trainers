An example of training a Llama model with huggingface dataset.

All shell scripts are in the "scripts/" folder. For example, you can first run:
```shell
cd examples/easyllama/
bash scripts/prepare_pretrain.sh
```
to create the pretrain dataset, and run:
```shell
bash scripts/pretrain_ddp.sh
```
to pretrain the llama model.

You can config the model, dataset, and training info in the "config/" folder.
