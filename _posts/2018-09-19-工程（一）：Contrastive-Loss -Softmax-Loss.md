---
layout:     post
title:      Contrastive Loss with Softmax Loss
subtitle:   Classification
date:       2018-09-19
author:     Shaozi
header-img: 
catalog: true
tags:
    - Classification
    - Deep Learning
---

任务背景：用户上传的图像中的商品进行分类，将其划分为12大类。
现有模型：Resnet-50
初步计划：在Softmax Loss的基础上增加Contrastive Loss进行约束
平台：linux, caffe

作为一个caffe新手，首先去调研了一下Contrastive Loss的作用和caffe下的使用方法。

首先该loss的定义可以参考[CSDN](https://blog.csdn.net/autocyz/article/details/53149760)这篇文章，可以说是很简洁了，公式如下。该loss被广泛的使用在孪生网络中，关于孪生网络，可以参考[知乎](https://zhuanlan.zhihu.com/p/35040994)，写的还挺有意思的。
![ ](https://upload-images.jianshu.io/upload_images/11609151-d23ca15ef8cbfd4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
背景知识准备完成，那么说一下为什么要将softmax loss和contrastive loss结合，其实很简单，就是想拉开不同类别之间的距离缩小相同类别之间的距离。其实Softmax loss自身也有类似的效果。但是由于样本数量巨大，故专门使用一个loss来强化该功能。使样本的特征空间分布更为稀疏。

Caffe中提供了contrastive loss的实现，具体的使用方法参照官方教程：[Siamese Network Training with Caffe](http://caffe.berkeleyvision.org/gathered/examples/siamese.html)。这里把该教程中的prototxt文件单独拿出来解释一下：
```c
layer {
    name: "loss"
    type: "ContrastiveLoss"
    contrastive_loss_param {
        margin: 1.0
    }
    bottom: "feat"
    bottom: "feat_p"
    bottom: "sim"
    top: "loss"
}
```
这里使用了一个参数margin，这里对应于公式中的margin。前两个bottom为两流的特征，sim为公式中的y，caffe源码中只考虑0和1（实现上其他整数值也可以）。
这样使用该loss的时候是不带类别标签的，只需要知道输入的两张图片是否为同一类即可。
caffe的样例的实现方法为：
```c
layer {
  name: "pair_data"
  type: "Data"
  top: "pair_data"
  top: "sim"
  include { phase: TRAIN }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/siamese/mnist_siamese_train_leveldb"
    batch_size: 64
  }
}
layer {
  name: "slice_pair"
  type: "Slice"
  bottom: "pair_data"
  top: "data"
  top: "data_p"
  slice_param {
    slice_dim: 1
    slice_point: 1
  }
}
```
可以发现，图片已经预处理为图像对，data层得到y和图像对，然后使用slice层将图像对生成两张图片，分别送入两路孪生网络。这样做的好处是可以保证训练样本总量中相同对和不同对的数量和比例，但是由于本任务的样本数量巨大。本来网络训练就已经很耗时了。如果要构建图像对，那么训练数据量至少要翻倍。另外图像增强等方法的使用问题也需要解决。

由于项目时间紧，而且该实现也属于尝试性质，考虑到类别数量不大。计划采用全随机的方式来生成图像对具体的实现方案如下：
——使用两个data层，分别读取一组图片样本，每一个图片样本包括他的图片和类别标签（0~11）。分别送入两路共享权重的resnet网络，分别接两个softmax loss。修改Contrastive Loss的源码，将输入改为4个，分别为两路的特征，和两路的标签，标签相同对应y=1，标签不同对应y=0。在原始的分类网络的特征输出上分别加上两个全连接层（共享权重）输出一个256维的向量，计算Contrastive Loss。
新的带分类的孪生网络的写法可以参考上面提到的官方的教程。Contrastive Loss代码的修改可以参考[博客](https://blog.csdn.net/zllljf/article/details/80970557)，具体而言就是将源码中所有如下判别：
```c
if (static_cast<int>(bottom[2]->cpu_data()[i]))
```
修改为：
```c
if (bottom[2]->cpu_data()[i] == bottom[3]->cpu_data()[i])
```
即可。当然记得增加相应的变量和判别。另外GPU的实现代码也需要修改。方法类似。
现在我们有三个loss，由于两个softmax是等价的，故weight设为0.5即可。考虑到Contrastive Loss中找到相同类的不高，暂时没有给他分配太高的权重，权重设为0.1。
个人认为比较合适的权重应该参照softmax收敛时的loss的大小a。因为Contrastive Loss最大值基本为1（没有相同类）那么Contrastive Loss的weight应该设为a*1，这样保证两个loss在总loss中的比重相同。当然也可以依据此基准进行调整。
