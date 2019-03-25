---
layout:     
title:      基于tripletloss的特征模型
subtitle:   Embedding
date:       2019-03-15
author:     Shaozi
header-img: 
catalog: true
tags:
    - Embedding
    - Deep Learning
---

任务背景：商品以图搜图
现有模型：SE-Resnet 50\*4d
初步计划：实现一个基于triplet loss的特征模型，并进一步考虑多级商品信息。
平台：linux, pytorch

图搜算法中存在一个问题：搜到的相关的图片存在大类之间的差异。希望通过加入多级的信息，优化特征分布，目标如下图：
![多级特征分布](https://i.loli.net/2019/03/15/5c8b5c5f978b4.png)

这里计划参考[Embedding Label Structures for Fine-Grained Feature Representation](https://arxiv.org/abs/1512.02895)
故需要首先构建一个基于triplet loss的embedding模型。

模型训练和评价的工程文件包含一下几个部分：
——root
  ——datasets（包含数据集转换部分的代码，这里不做介绍）
  ——imagedata（包含训练和测试所用的原始图片数据和标注）
  ——res（缓存文件）


