---
layout:     post
title:      基于mmdetection检测模型学习（1）
subtitle:   mmdetection
date:       2021-01-05
author:     Shaozi
header-img:
catalog: true
tags:
- mmdetection
- detection
---

## 前言

最近半年开始频繁的接触工业质检相关的项目，大多数都要用到检测技术。基于之前的技术栈，起初是在detectron2框架下进行算法的训练。随着项目的增多，发现mmdetection的可玩性更高。所以逐渐迁移到这个平台。接近年底，很多项目都接近尾期，闲下来有时间好好研究一下mmdetection的细节。受到[mmdetection-mini](https://github.com/hhaAndroid/mmdetection-mini)的启发，决定将我学习的过程记录下来，所以开了一个新的系列。

我打算以一个项目为切入点，对mmdetection的各个实现细节做一个详细的探究。这里采用工业质检中公开的一个比较基础的小数据集[NEU-DET](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)数据集作为学习的数据集，采用以Faster-RCNN模型为基础进行训练。

## 先跑起来