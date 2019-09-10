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

代码能够复现论文中的性能，但是在我的目标数据集上表现欠佳，由于使用了faiss所以代码不怎么支持GPU并行，训练效率较低。

## Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning

这篇论文角度新颖，通过归纳loss计算中梯度规律，对不同的度量学习loss进行了整理，将形式各样的loss通过同样的形式进行表达。并且分析了不同loss在优化过程中的相同点和不同点。并且通过归纳和整理，提出了一种更为全面的Multi-Similarity Loss。在多个实验集上都有很好的实验效果。

本篇文章对不同的度量loss有较为全面的整理，并且对不同loss之间的区别和联系有较为深入的分析，很适合用来学习。

Code主页已放出，但还未开源。实验实现表述较少，复现难度较大。

## 补充：No Fuss Distance Metric Learning using Proxies

这篇论文概括起来很简单，在计算loss的时候不再挑选样本，而是从一个池子中挑选出样本的代理来进行比较。这个池子是不断学习更新的。有点像电影《铁甲钢拳》，PK的时候不是用真人，而是使用一个替代品（机器人）。这里代理的数量决定了最后的性能，论文中给的建议是大于0.5的图片类别数。

我这里使用了一个非官方复现的版本：[Git][1]

同样这版论文不支持分布式训练。我对代码进行了修改。原版代码和修改后的代码都可以复现论文中的性能。惊奇的发现这个算法的效率奇高，基本几分钟就能出现很好的结果了。（GPU为v100）

于是我在我的目标数据集上进行了测试。然后问题来了，由于目标数据集包含了两万多类，按照按照代码中的实现，代理样本的数量为类别数，在计算时降计算一个大小为2w\*embedding的tensor。直接超显存。不过个人觉得有优化的空间，因为一个2048-2048的全连接层，tensor的大小也是差不多这个量级的，为什么不怎么吃显存呢？

下面是code reading部分：
loss input：
```python
loss = criterion(m, y.cuda())
```
其中`m`为embedding，batchsize为256，dim为64，所以是一个256\*64的tensor。
`y`为label，64\*1
Loss函数如下：
```python
class ProxyNCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, smoothing_const = 0.0, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(int(nb_classes*0.51), sz_embed) / 8)
        self.smoothing_const = smoothing_const

    def forward(self, X, T):

        P = self.proxies
        P = 3 * F.normalize(P, p = 2, dim = -1)
        X = 3 * F.normalize(X, p = 2, dim = -1)
        # 改为 X = 3 * X
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = self.smoothing_const
        )

        # cross entropy with distances as logits, one hot labels
        # note that compared to proxy nca, positive not excluded in denominator
        loss = torch.sum(- T * F.log_softmax(D, -1), -1)

        return loss.mean()
```
其中`proxies`为代理数量，这里使用了`torch.nn.Parameter`。将一个普通的tensor转换为可训练的tensor。`smoothing_const`，暂时不管是干嘛的。可以看到`proxies`最初是随机生成的。在forward中对其进行了L2 norm。很奇怪的一点是为什么要\*3。另外embedding `X`输出的时候已经经过一次L2 norm了，这里应该可以删去。虽然再计算一遍数值也不会变，但是占用了计算资源。

接下来将`X`和`P`cat一下送入函数`pairwise_distance`:
```python
def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )
    # 修改为： pairwise_distances_squared = torch.mm(a, torch.t(a))
    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )
    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )
    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )
    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
    return pairwise_distances
```

这里第一步就把我看懵了……
(省略中间过程)
整个函数应该和`D = 18 - 2 * torch.mm(X, P.t())`等价。

优化完这里，计算loss就轻量很多了。接下来可以开始训练目标数据集了。但是似乎不收敛…..(未完待续)


[1]:	https://github.com/dichotomies/proxy-nca

[image-1]:	http://ww1.sinaimg.cn/large/c310f833ly1g4nn6ebjlmj20bw07edhf.jpg
[image-2]:	http://ww1.sinaimg.cn/large/c310f833ly1g4nomt96czj209k03kwer.jpg
[image-3]:	http://ww1.sinaimg.cn/large/c310f833ly1g4nrxraovmj20of07bgpi.jpg