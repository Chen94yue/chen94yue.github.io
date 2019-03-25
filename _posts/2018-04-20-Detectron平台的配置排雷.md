---
layout:     
title:      Detectron平台的配置排雷
subtitle:   Detection
date:       2018-04-20
author:     Shaozi
header-img: 
catalog: true
tags:
    - Detection
    - Detectron
    - Deep Learning
---

Detectron平台是什么，做检测的各位小伙伴应该不用再做过多的介绍了。刚出的时候配置过一次该平台，没有出现什么问题。caffe2和pytorch合并后似乎再搭建该平台，就有点坑了。遇到了不少问题，调试了好几天，因为比较菜，每次环境乱了都重装一遍系统，最后终于成功了，网上很多解决方法都不太新，现将我遇到的问题总结如下。
**因为是全部调完了之后才写着部分总结，所以可能没办法贴出报错的原始输出了**
这个问题很简单，按照官网上的教程，就不会有任何问题。但是有以下几点需要注意。
第一，**建议使用ubuntu16.04**，我开始使用的16.04，似乎该版本的caffe2对detectron的支持有问题。14.04下GPU的支持可能会有很多错误，当时解决了好多，但是最后也没有通过。反正秉着一条信念，如果你的做法和官网上的做法差距很大的话，多半是你的做法有问题。
第二，detectron**不能使用conda安装的caffe2**。这一点其他的教程中也有提到，因为detectron太新，conda下集成的caffe2似乎不能完全满足detectron的要求。在我这里问题出在libnccl.2.so上，conda下可以提供的nccl最新的版本是1.3.4，配置后在库中的文件是libnccl.1.so，而detectron需要最新的，我找了很多方法，没有找到得到libnccl.2.so的方法。我甚至建议先卸载掉conda再配置环境，因为python路径的混乱可能会导致caffe2 make时候的混乱。
第三，**不要头铁非要使用cudnn v5**。虽然在caffe2的官方教程上建议使用v5或者v6，甚至指令示意都是给的v5的，但是，在新版的caffe2中，建议使用v6的版本，最烦的是你在make他的时候，他会不停的提示你使用v6的版本。因为nvidia的官网，cudnn的下载页面屏蔽了大陆的ip，所以现在去下cudnn会很麻烦，需要翻墙，在此提供一个百度云的分享吧，如果无效了可以在下面评论，或者微信联系我更新。[链接](https://pan.baidu.com/s/1l-ikg75jDeO0Vhfr2cFx7Q)

基本上避开上诉三个雷区，按照官网的教程，先安装cuda8.0，cudnn v6，caffe2。最后按照detectron的教程安装就可以了。注意上述一段话的重点是“按照官网的教程”，特别是cuda8.0和cudnn v6，网上的教程千奇百态由于太多了，所以不打算写一个专门的教程来介绍。但是实际上五行指令就可以解决问题。这里简单说一下吧。
安装cuda前不需要做任何准备，前提你是台式机，没有cpu上坑爹的集成显卡。现在是一个刚刚装好的ubuntu16.04系统。你去cuda的官网，下载需要的cuda版本。以我使用的为例，下载好之后，在目录下打开终端，依次输入：
```shell
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```
安装会自动安装好cuda和显卡驱动。**不需要先装好显卡驱动再装cuda**。
cudnn的安装就更简单了，以v5为例吧，按照caffe2官网的教程即可：
```shell
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```
当然如果你是自己下载的**只需要后面两行即可**

接下来，按照caffe2官网的教程配置caffe2了。这里先保证**系统中只有一个python**吧，如果有装conda的，conda中的python路径写在系统变量中的，如果出现了问题，可以去参考其他教程的方法。或许可以解决问题，但是我遇到的类似问题按照其他教程说法尝试之后并没有解决。如果系统中只有一个python，那么按照官网的教程，应该可以直接全部通过。
首先安装相关环境：
```
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      python-pip                          
sudo pip install \
      future \
      numpy \
      protobuf
```
接下来git一下caffe2，并且make它
```shell
# Clone Caffe2's source code from our Github repository
git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch
git submodule update --init

# Create a directory to put Caffe2's build files in
mkdir build && cd build

# Configure Caffe2's build
# This looks for packages on your machine and figures out which functionality
# to include in the Caffe2 installation. The output of this command is very
# useful in debugging.
cmake ..

# Compile, link, and install Caffe2
sudo make install
```
最后用两个小测试检测一下caffe2是否配置好了：
```shell
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```
输出sucess即为成功
```shell
python caffe2/python/operator_test/relu_op_test.py
```
不报错即为成功

这两步我似乎碰到了一点小问题，好像是import找不到文件的。如果你是严格按照教程来的，现在应该不存在没有安装的包。所以用下面两条指令处理一下：
```shell
cd /usr/local/lib
sudo ldconfig
```
再运行上面两条测试指令，应该没错了。那么恭喜，基本没有问题了。
detectron的配置，还是按照官网来，在前面pip包的时候，会报failed，错误是bdist_whell相关的。但是包可以正常安装，我尝试解决了一下，没有解决，因为折腾太久了，反正现在也可以用，就没有再纠结这个问题。
后面按照配置来就可以了。基本没有什么问题。如果有问题就是环境不对，在网上找解决方法吧。

下一篇写detectron平台的一些简单的使用。caffe2我也是刚刚接触，如果有必要的话，再更新一个caffe2的入门系列吧。




