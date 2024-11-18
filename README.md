# tokenmix2

## 各个文件夹的功能

P.S. 如果目录中不存在下面提到的文件夹，则需要自己使用 mkdir 来创建

| dir | functionality |
| --- | ------------- |
| ckp | 保存模型的checkpoint |
| config | 保存训练配置 |
| data | 保存训练数据，此文件夹应该软连接到hdd上，来减少ssd的开销 |
| data_cache | 保存数据preprocess中的一些关键信息，使得二次加载时节省时间 |
| log | 保存训练的日志 |
| LongBench | 一个长文本qa的benchmark |
| mmlu | MMLU benchmark |
| test_draft | 测试draft模型的基准benchmark，在这个小benchmark上表现好，大概率能够在其他上表现好 |
| test_lmeval | 测试在各种natural language understanding任务上的表现，包括glue, superglue, arc, openbookqa等等 |
| test_longbench | 测试在longbench benchmark上的性能 |
| test_ppl | 测试language modeling perplexity |
| test_ruler | 测试在ruler数据集上的表现 |
| test_walltime | 测试模型的latency |
| tokenmix2 | 最主要的文件夹，保存了核心代码 |
| train | 与训练有关的代码 |
| train_cache | 保存训练数据的地方 |
| train_results | 训练结果 |


## 安装

```
conda create -n genacc -y python==3.10
conda activate genacc

git clone https://github.com/wenhaoli-xmu/genacc.git --branch dev_lwh
git clone https://github.com/wenhaoli-xmu/lm-corpus.git
git clone https://github.com/wenhaoli-xmu/lm-profiler.git

cd lm-corpus
pip install -e .
cd ..

cd lm-profiler
pip install -e .
cd ..

cd genacc
pip install -e .
pip install -r requirements.txt
```


## 大文件迁移

**从MAC85迁移data文件夹**

MAC85地址：10.24.116.85: 1172

具体来说，就是data文件夹太大，所以没有随着git上传，因此git clone之后需要将data文件夹拷贝过来。
data文件夹的目录在 lwh@10.24.116.85:/home/lwh/token-mix-3/data，可以自行在85上创建账号，然后用scp命令拷贝，
也可以联系我拷贝。

**从MAC85迁移train_cache文件夹**

这个文件夹保存了已经准备好的训练数据（这个代码是逐层训练的，因此train_cache中保存了每一层的激活值作为输入）。
如果不想拷贝这个，也可以选择使用以下命令自己生成：

```bash
python train/train_prepare.py --env_conf train/genacc19-10.json
```

生成之前需要删除train_cache下已有的所有文件

## 训练

在训练之前要先保证大文件迁移完成，不然没有训练数据

启动训练的命令是

```bash
python train.py --env_conf train/genacc19-10.json --gpus[0,1,2,3,4,5,6,7] --num_layers 32
```

## 合并checkpoint

由于训练是逐层进行的，因此每一层的训练结果会被单独地保存下来。训练完成之后需要将每一层的权重合并成为一个完整的权重

具体而言运行下面的命令即可：

```bash
python train_results/convert.py genacc19-10.json genacc19-10.pth
```

convert.py会自动在train_result下搜索genacc19-10的训练结果，并将其汇总到 ckp/genacc19-10.pth 中

## 测试

**排序误差**

```bash
python test_draft/test_other.py --env_conf test_draft/genacc20.json 
```

注意这里 genacc19-xxx 前缀的 config 需要使用 genacc20 config 来进行测试。在genacc20中，有如下的部分：

```json
"model": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "model_dtype": "fp16",
    "model_method": "genacc20",
    "model_structure": null,
    "save_ckp": "ckp/genacc19-9.pth",
    "load_ckp": null,
    "config": "config/genacc15.json",
    "device_map": null
}
```

其中save_ckp可以修改为自己训练得到的那个checkpoint文件。如果设置为 "null"（一定是字符串null，和load_ckp不同，这里算个bug但是一直没改），则表示不加载任何checkpoint。
config字段表示配置，里面包含剪枝率等等重要的参数，如下所示：

```json 
{
    "fix_layers": [0,1],

    "draft_kwargs": {
        "enable": true,
        "mask_out": 0.98,
        "bench_mark": false
    },

    "enable_lora": false,
    "lora_kwargs": {
        "lora_rank": 128,
        "lora_alpha": 512,
        "lora_dropout": 0.1
    }
}
```

其中maskout代表剪枝率，当前为98%，enable表示是否启用剪枝，启用为true，不启用为false。bench_mark为结果可视化，可以启用，但是会比较慢。


具体测试的数据集在test_draft/eval.json中，如下：
```json
[
    {
        "task_type": "perplexity",
        "task_name": "pg19.test.128k",
        "num_instance": 1,
        "truncation": 2048
    },
    {
        "task_type": "perplexity",
        "task_name": "pg19.test.128k",
        "num_instance": 1,
        "truncation": 4096
    }
]
```
