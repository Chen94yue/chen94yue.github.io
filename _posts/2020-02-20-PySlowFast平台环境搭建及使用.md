---
layout:     post
title:      PySlowFast平台环境搭建及使用
subtitle:   PySlowFast
date:       2020-02-20
author:     Shaozi
header-img:
catalog: true
tags:
- Video Recognition
- Video Action Classification
---

### 1.环境搭建

[SlowFast](https://github.com/facebookresearch/SlowFast) 平台是Facebook近期开源的视频识别平台。这里对该平台的环境搭建和使用进行记录和总结。

由于平台依赖Detectron2，Detectron2需要Pytorch 1.3以上版本，Pytorch1.3以上版本需要CUDA10.1及以上版本，所以在CUDA10.0及以下的机器上无法使用，建议升级CUDA及对应的显卡驱动。

如果之前并没有使用过FFmpeg等视频库以及Detectron2，那么PySlowFast的开发环境搭建可以参考代码主页提供的[INSTALL.md](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md)。如果已经安装了相关的环境，不想破坏现有环境，或者PySlowFast需要的库版本和当前系统版本冲突，建议使用docker容器。这里提供一份dockerfile：

```Dockerfile
FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install vim libsm6 libxrender1 libxext-dev

WORKDIR /opt

RUN cd /opt && \
    pip install 'git+https://github.com/facebookresearch/fvcore' && \
    pip install simplejson && \
    conda install av -c conda-forge && \
    pip install opencv-python && \
    pip install cython && \
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN cd /opt && \
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo && \
    pip install -e detectron2_repo

RUN cd /opt && \
    git clone https://github.com/facebookresearch/slowfast && \
    export PYTHONPATH=/opt/slowfast/slowfast:$PYTHONPATH && \
    cd slowfast && \
    python setup.py build develop
```

接下来，开始准备数据集，PySlowFast平台目前支持AVA和Kinetics两个数据集，并且都提供了下载和准备的[说明和脚本](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md)，Kinetics数据集较大，而且似乎需要从youtube上面下载视频，由于“不可抗”的原因，这里使用AVA数据集进行测试。数据集的下载及处理参考上面的连接就行。这里不再赘述。

AVA数据集准备好之后，我们就可以直接调用脚本[tools/run_net.py](https://github.com/facebookresearch/SlowFast/blob/master/tools/run_net.py)来测试一下性能了。

```
python tools/run_net.py \
  --cfg configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml
```

这里就以该脚本详细分析一下代码构成

### 2.代码构成

run_net.py功能为读取cfg，然后依据cfg的设置调用test_net.py或者train_net.py。run_net.py的导入参数如下：


|参数|作用|
|:-:|:--|
|shard_id|分布式训练使用的节点编号|
|num_shards|分布式训练使用的节点总数|
|init_method|分布式训练使用的通讯方式，[详见](https://chenyue.top/2019/03/28/%E5%B7%A5%E7%A8%8B-%E5%9B%9B-Pytorch%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/)。|
|cfg|指定配置文件的路径|
|opts|可覆盖cfg文件中的参数|

#### （1）读取config

SlowFast的config设置基于facebook的fvcore库实现，具体使用可以[参考](https://github.com/facebookresearch/fvcore)

各种设置的说明详见['slowfast/config/defaults.py'](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/config/defaults.py).

具体而言config的读取顺序为：

“slowfast/config/defaults.py” ➡️ cfg_file（--cfg指定的路径下的配置文件）➡️  opts (--opts下指定的额外的配置)

-------
*在读取完cfg文件之后，顺便把checkpoint的路径生成了。这里调用了‘slowfast.utils.checkpoint’下的make_checkpoint_dir。目前slowfast.utils.checkpoint支持的主要功能包括：*

|方法名|功能|细节|
|:-:|:-:|:--|
|make_checkpoint_dir|生成checkpoint的保存路径|只在分布式训练的主节点上生成|
|get_last_checkpoint|找到当前保存路径下最新的checkpoint文件||
|save_checkpoint|保存模型训练的中间参数|不仅仅保存了模型文件，还保存了配置文件和当前的optimizer情况，值得学习|
|load_checkpoint|读取模型文件|支持将2d卷积读取为3d|

-------

（2）Test

这里直接使用训练好的模型，所以先看一下test的逻辑。对应的配置文件为：

```python
cfg.TRAIN.ENABLE = False
cfg.TEST.ENABLE = True
```

因为运行的机器有四张显卡，所以有：

```python
cfg.NUM_GPUS = 4
```

因为这样的设置会调用如下的方法开始测试：

```python
torch.multiprocessing.spawn(
    mpu.run,
    nprocs=cfg.NUM_GPUS,
    args=(
        cfg.NUM_GPUS,
        test,
        args.init_method,
        cfg.SHARD_ID,
        cfg.NUM_SHARDS,
        cfg.DIST_BACKEND,
        cfg,
        ),
    daemon=False,
)
```

**去看slowfast/utils/multiprocessing.py下的run方法，会发现输入的参数多了一个local_rank，这里就需要考虑torch.multiprocessing.spawn的实现了，在torch.multiprocessing.spawn中调用的是multiprocessing的Process方法建立进程，但是建立进程时调用的是_warp函数，该函数使用的参数在原先的args的基础上增加了一个i，这个i就是local_rank，所以在我们使用的时候，该参数不用专门设定，代码会依据nprocs的数量自动生成。[参考](https://pytorch.org/docs/1.3.1/_modules/torch/multiprocessing/spawn.html#spawn)**

下面开始测试，这里跳转到[test_net.py](https://github.com/facebookresearch/SlowFast/blob/master/tools/test_net.py)

设置随机数种子点，设置log什么的就略过了，直接看模型的初始化。这里调用了build_model方法。跳转到[build.py](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/build.py)

这一部分的操作比较秀，使用fvcore.common.registry构建了一个修饰器，依据模型的名称调用[video_model_builder.py]下的对应的模型类构建模型。这里如果要新增自己的网络，只需要在video_model_builder.py中定义相关的类就行。

接下来调用load_checkpoint方法加载训练好的模型参数，这里需要通过cfg.TEST.CHECKPOINT_FILE_PATH指定模型参数的位置。或者指定模型输出的位置，程序会默认加载最新的模型。如果不指定测试的模型位置，会从训练的模型保存位置加载模型，如果完全不指定模型位置，会随机初始化网络。

接下来就是加载数据，并开始测试了。

未完待续。。。

