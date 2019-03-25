---
layout:     
title:      Embedding Label Structures for Fine Grained Feature R
subtitle:   Fine-grained
date:       2018-05-19
author:     Shaozi
header-img: 
catalog: true
tags:
    - Fine-grained
    - Deep Learning
---

前言：
关于Fine-grained的工作之前没有接触过。所以打算从六篇论文开始，也算是一个学习的过程。因为没有基础，可能在文章理解上会存在偏差，如果文章中有什么问题，欢迎指正。
目录：
[2016[CVPR]-Embedding Label Structures for Fine-Grained Feature Representation](https://arxiv.org/pdf/1512.02895)
[2016[CVPR]-Learning Deep Representations of Fine-Grained Visual Descriptions](https://arxiv.org/pdf/1605.05395)
[2016[CVPR]-Part-Stacked CNN for Fine-Grained Visual Categorization](https://arxiv.org/pdf/1512.08086)
[2017[CVPR]-Low-rank Bilinear Pooling for Fine-Grained Classification](https://arxiv.org/pdf/1611.05109)
[2017[CVPR]-Mining Discriminative Triplets of Patches for Fine-Grained Classification](https://arxiv.org/pdf/1605.01130)
[2017[ICCV]-Efficient Fine-grained Classification and Part Localization Using One Compact Network]()
[2017[ICCV]-Fine-grained recognition in the wild A multi-task domain adaptation approach]()
[2017[TMM]-Diversified visual attention networks for fine-grained object classification]()

#论文：Embedding Label Structures for Fine-Grained Feature Representation
本文主要的创新点是提出了一种细粒度的结构化的特征表示，用于在不同的level关联和区分相近的图片。
关键点有2：
1.一种多任务的学习框架被设计用于学习细粒度的特征表示，通过联合优化分类和相似性约束。
2. Triple loss的使用将多级的相关性和标签结构无缝的融合。

对于现有的方法，作者指出，尽管Triple loss能够很好的区分类的实例，但是会照成分类准确度的下降，并且增加训练收敛的时间，此外之前的工作没有提出标签结构这样的框架，而这个框架对于在不同的级别定位图像至关重要。
本文的创新点主要有两个，第一个是提出了一种多任务的学习框架，同时用雕了分类损失和相似性损失。第二个创新点是本文设计了一种嵌入标签结构，比如层级或者属性。用于区分图片。

##论文方法
传统的用于分类的方法使用softmax loss来解决分类问题，但是这种loss往往会丢失类中的差别。为了缓解这个问题，本文采用了一个多任务的学习方法，引入了triplet loss。在文中水了一页softmax和triplet loss的介绍，我就不展开写了。如果对这两个loss的原理有不清楚的地方，可以通过百度或者google搜到很多讲解的文章。

为了结合这两种loss，网络被设计成了如下所示的三流的结构，这三流是共享参数的，但是输入的不同的样本，对应triplet loss的三个部分。
![算法流程图](https://upload-images.jianshu.io/upload_images/11609151-e367621eb6193f8d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来文章介绍了第二个创新点——标签结构，这里文章中提出了两种标签结构，一种是基于分层的，一种是基于属性的。基于分层的结构如下图所示：
![分层结构](https://upload-images.jianshu.io/upload_images/11609151-a2cd89212fba3997.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
为了在训练中实现这种分层结构，本文将传统的triplet loss增加了一项，该项表示那些和挑选的样本在父类中属于同一类的个体。然后扩写了triplet loss的函数，将其改为：
![qurdruplets函数](https://upload-images.jianshu.io/upload_images/11609151-e8559434a6ea4a0e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
按照这样的格式，这个公式可以扩写为x个level，这样由x子的triplets函数组成

出了上面的分层的结构，细粒度的物体之间可以共享一些属性，如下图所示：
![共享属性](https://upload-images.jianshu.io/upload_images/11609151-ab31f5ed4a743d90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
不同于上面的分层的结构，在讨论属性时，不同类别的物体可能共享相同的属性，所以，不能再像之前一样通过扩增loss函数来解决问题。这里再loss函数的常熟间隔上做文章，将m改为：
![m](https://upload-images.jianshu.io/upload_images/11609151-9c95896ab8a0bdc7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中括号中部分为1减属性的IoU，这样，两个物体共享的属性越多，他们在计算loss时需要考虑的间隔越小。
实验结果验证了文章的方法。在此不再详细介绍，感兴趣的同学可以去看原文。
