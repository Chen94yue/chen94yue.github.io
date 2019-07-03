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

 
# Paper List

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

# 数据集及评价指标：

## CUB-200-2011

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

## CAR196

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

## SOP

|Method|R@1|R@10|R@100|
| 3.RLL-(L,M,H) | 79.8 | 91.3 | 96.3 |
| 5.SCHE | 77.6 | 89.1 | 94.7 |
| 6.MS | 78.2 | 90.5 | 96.0 |
| 7.FastAP | 75.8 | 89.1 | 95.4 |
| 9. DE-DSP (N-pair) | 68.9 | 84.0 | 92.6 |
| 10. DCES | 75.9 | 88.4 | 94.9 |
| 13. RRM | 69.7 | 85.2 | 93.2 |
| 14. HDML | 68.7 | 83.2 | 92.4 |

## In-shop

|Method|R@1|R@10|R@20|R@30|
| 5.SCHE | 91.9 | 98.0 | 98.7 | 99.0 |
| 6.MS | 89.7 | 97.9 | 98.5 | 98.8 |
| 7.FastAP | 90.9 | 97.7 | 98.5 | 98.8 |
| 9. DE-DSP (N-pair) | 78.6 | 93.8 | 95.5 | 96.2 |
| 10. DCES | 85.7 | 95.5 | 96.9| 97.5 |