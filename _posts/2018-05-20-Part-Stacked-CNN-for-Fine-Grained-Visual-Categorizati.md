---
layout:     post
title:      Part-Stacked-CNN-for-Fine-Grained-Visual-Categorizati
subtitle:   Fine-grained
date:       2018-05-20
author:     Shaozi
header-img: 
catalog: true
tags:
    - Fine-grained
    - Deep Learning
---

目录：
[2016[CVPR]-Embedding Label Structures for Fine-Grained Feature Representation](https://arxiv.org/pdf/1512.02895)
[2016[CVPR]-Learning Deep Representations of Fine-Grained Visual Descriptions](https://arxiv.org/pdf/1605.05395)
[2016[CVPR]-Part-Stacked CNN for Fine-Grained Visual Categorization](https://arxiv.org/pdf/1512.08086)
[2017[CVPR]-Low-rank Bilinear Pooling for Fine-Grained Classification](https://arxiv.org/pdf/1611.05109)
[2017[CVPR]-Mining Discriminative Triplets of Patches for Fine-Grained Classification](https://arxiv.org/pdf/1605.01130)
[2017[ICCV]-Efficient Fine-grained Classification and Part Localization Using One Compact Network]()
[2017[ICCV]-Fine-grained recognition in the wild A multi-task domain adaptation approach]()
[2017[TMM]-Diversified visual attention networks for fine-grained object classification]()

摘要：本文提出了一种部件堆结构的CNN网络，用于得到细粒度分类图的明确解释。结构包含一个部件定位的全卷机网络，和一个双流的分类网络，用于同时编码部件区域和全局区域。并且在部件和全局之间采用一种共享的机制。

实际上从文章的introduction中的描述可以看出，本文做的还是一个细粒度图像分类的任务。摘要中提到的解释，应该指的是部件的分类。算法的整体框架如下：
![算法网络框架](https://upload-images.jianshu.io/upload_images/11609151-eb5ce274c4fb0636.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
结构显而易见，对于输入的一张待分类图片，将其送入特征提取网络，得到全局的体格特征，经过上采样，然后分别送入相同结构的特征提取网络，提取深度特征，和一个FCN网络用于产生部件区域，然后经过一个类似ROI pooing的操作，得到部件区域的深度特征，将其和整体的特征串联起来，送到后续的全连接网络做分类的操作。

部件区域检测方面，论文借鉴了人体姿势估计和语义分割的工作，将FCN网络用于关键点预测。具体结构如下：
![FCN结构](https://upload-images.jianshu.io/upload_images/11609151-db24c974f3e1431a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这里，在通过主干网络之后，通过三个1*1的卷积，将channel数降为M+1，M为部件的个数，1是背景。在计算loss时，每一个特征图对于一个关键点的标注图，计算全图的softmax loss。公式如下：
![公式1](https://upload-images.jianshu.io/upload_images/11609151-df1eac8c98148e88.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

特征网络这边采用了参数共享的策略，减少了网络的参数量。在得到部件的位置（点）后，取以该点为中心的6*6的区域作为部件的特征区域提取出来。将所有的部件区域的特征和整体的特征串联起来，这样形成了一个6*6*（256+32M）的特征图，经过一个1*1的卷积降维为6*6*32维之后做后面细粒度的分类（FC）。
