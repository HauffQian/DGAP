## Regression Task

This example implements regression task for discriminator. 
It involves two directories representing the benchmark data generation process.T

It supports 
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) or customizing by yourself
* assigning a classfication model from [transformers](https://github.com/huggingface/transformers) 
* multi-host running

### Requirement

Install Redco
```shell
pip install redco==0.4.16
```

### Usage

```shell
python main.py \
    --dataset_name sst2 \
    --model_name_or_path roberta-large \
    --n_model_shards 2
```
* `--n_model_shards`: number of pieces to split your large model, 1 by default (pure data parallelism). 

See `def main(...)` in [glue_main.py](glue_main.py) for all the tunable arguments. 


## Acknowledgments

Part of training code is based on resources from the following project:

- [redco](https://github.com/tanyuqian/redco)
