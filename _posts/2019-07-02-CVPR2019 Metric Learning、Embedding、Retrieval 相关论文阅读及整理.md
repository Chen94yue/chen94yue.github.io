---
layout:     post
title:      CVPR2019 Metric Learning、Embedding、Retrieval 相关论文阅读及整理
subtitle:   Paper reading
date:       2019-07-01
author:     Shaozi
header-img:
catalog: true
tags:
	- Metric Learning
	- Embedding
	- Retrival
---

 
## Paper List

这里从CVPR2019收录的论文中挑选了出了标准的度量学习相关的论文，相近领域以及一些特定应用场景的论文被剔除了。总共有十四篇。

1. A Theoretically Sound Upper Bound on the Triplet Loss for Improving the Efficiency of Deep Distance Metric Learning
2. End-to-End Supervised Product Quantization for Image Search and Retrieval
3. Ranked List Loss for Deep Metric Learning
4. On Learning Density Aware Embeddings
5.  Stochastic Class-based Hard Example Mining for Deep Metric Learning
6. Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning
7.  Deep Metric Learning to Rank
8. Learning Metrics from Teachers: Compact Networks for Image Embedding
9. Deep Embedding Learning with Discriminative Sampling Policy
10. Divide and Conquer the Embedding Space for Metric Learning
11. Unsupervised Embedding Learning via Invariant and Spreading Instance Feature
12. Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning
13. Deep Asymmetric Metric Learning via Rich Relationship Mining
14. Hardness-Aware Deep Metric Learning

## 数据集及评价指标：

这里对文章中的实验结果进行了总结，方便对比，但是由于不同的方法的实验条件不同，所以不能完全依靠实验结果来判断算法的好坏。

### CUB-200-2011

|Method|R@1|R@2|R@4|R@8|
| 1. Discriminative | 51.43| 64.23 | 74.31 | 82.83 |
| 3.RLL-(L,M,H) | 61.3 | 72.7 | 82.7 | 89.4 |
| 5.SCHE | 66.2 | 76.3 | 84.1 | 90.1 |
| 6.MS | 65.7 | 77.0 | 86.3 | 91.2 |
| 9. DE-DSP (N-pair) | 53.6 | 65.5 | 76.9 | - |
| 10. DCES | 65.9 | 76.6 | 84.4 | 90.6 |
| 12. DSML | 51.6 | 54.9 | - | - |
| 13. RRM | 55.1 | 66.5 | 76.8 | 85.3 |
| 14. HDML | 53.7 | 65.7 | 76.7 | 85.7 |

### CAR196

|Method|R@1|R@2|R@4|R@8|
| 1. Discriminative | 68.31 | 78.21 | 85.22 | 91.18 |
| 3.RLL-(L,M,H) | 82.1 | 89.3 | 93.7 | 96.7 |
| 5.SCHE | 91.7 | 95.3 | 97.3 | 98.4 |
| 6.MS | 84.1 | 90.4 | 94.0 | 96.5 |
| 9. DE-DSP (N-pair) | 72.9 | 81.6 | 88.8 | - |
| 10. DCES | 84.6 | 90.7 | 94.1 | 96.5 |
| 12. DSML | 49.1 | 52.4 | - | - |
| 13. RRM | 73.5 | 82.6 | 89.1 | 93.5 |
| 14. HDML | 79.1 | 87.1 | 92.1 | 95.5 |

### SOP

|Method|R@1|R@10|R@100|
| 3.RLL-(L,M,H) | 79.8 | 91.3 | 96.3 |
| 5.SCHE | 77.6 | 89.1 | 94.7 |
| 6.MS | 78.2 | 90.5 | 96.0 |
| 7.FastAP | 75.8 | 89.1 | 95.4 |
| 9. DE-DSP (N-pair) | 68.9 | 84.0 | 92.6 |
| 10. DCES | 75.9 | 88.4 | 94.9 |
| 13. RRM | 69.7 | 85.2 | 93.2 |
| 14. HDML | 68.7 | 83.2 | 92.4 |

### In-shop

|Method|R@1|R@10|R@20|R@30| 
| 5.SCHE | 91.9 | 98.0 | 98.7 | 99.0 |
| 6.MS | 89.7 | 97.9 | 98.5 | 98.8 |
| 7.FastAP | 90.9 | 97.7 | 98.5 | 98.8 |
| 9. DE-DSP (N-pair) | 78.6 | 93.8 | 95.5 | 96.2 |
| 10. DCES | 85.7 | 95.5 | 96.9| 97.5 |

**下面按照以上统计的性能高低概括论文**
## Stochastic Class-based Hard Example Mining for Deep Metric Learning


![][image-1]
本文主要提供了一个度量学习中男例挖掘的方法，简单的讲就是度量学习和分类的结合。使用分类的结果来挑选计算loss的难例。只计算难例类中的难样本的loss。
方法上感觉不是很novel（也可能是我没看懂），但是性能很好。有几个Trick记录一下：
- 计算batch中的loss的均值时，只考虑大于0的。
- 使用了second-order pooling。
- 从第四个stage提特征。
- input的图像从224改为336
从文章提供的结果来看，后三个Trick的效果还是很明显的：
![][image-2]

实验结果还是比其他论文高太多了，没有公开代码。可能有坑

## Divide and Conquer the Embedding Space for Metric Learning

本文认为，在度量学习中，Embedding空间并没有被很好的利用。 所以提出了一个聚类加度量学习的框架，并且能够很好的应用于不同的算法下。
![][image-3]
这篇文章写的比较通俗易懂，属于很好实现的类型，而且提供了代码，性能相比于上一篇差别不大，很有借鉴价值。
仔细看来这一篇和上一篇很相近，从两个不同的角度优化了计算loss时样本的选择。其实都是在挑相近的难例。一个用聚类，一个用分类。可能有监督的分类任务更胜一筹吧。但是对于没有类别标签的任务，这一篇可能更有价值。
这一篇的聚类部分使用了k-means，不知道实际使用时速度如何，特别是在大规模数据的应用场景下。不过聚类是每一个epoch进行的工作，所以应该还好。如果知道大类别，能不能直接按类别进行划分了？

我决定实验一下。。。。。。。。

（未完待续）

[image-1]:	http://ww1.sinaimg.cn/large/c310f833ly1g4nn6ebjlmj20bw07edhf.jpg
[image-2]:	http://ww1.sinaimg.cn/large/c310f833ly1g4nomt96czj209k03kwer.jpg
[image-3]:	http://ww1.sinaimg.cn/large/c310f833ly1g4nrxraovmj20of07bgpi.jpg