---
layout:     post
title:      DetNet-A-Backbone-network-for-Object-Detection
subtitle:   Detection
date:       2018-04-20
author:     Shaozi
header-img: 
catalog: true
tags:
    - Detection
    - Deep Learning
---

这是清华和旷视的团队最新的研究成果。设计了一种专门用于检测的网络Detnet。文章首先指明了检测任务和分类任务的不同，这一点在FPN的论文中也已经提到。并且提到目前的方法大都采用增加额外的层来应对多尺度的问题。

目前的检测算法都是基于分类的主干网络（VGG16等）这样的设计存在三个问题：
1.为了完成检测任务需要增加额外的层
2.分类网络的大量下采样丢失了小目标的位置信息

因此设计了主干网络detnet用于解决这个问题。为了避免高分辨率特征图带来的计算复杂度和高额内存消耗，论文采用了一种 low complexity dilated bottleneck structure，在获得较高的感受野的同时保证了特征图分辨率足够的大。

论文在related works中介绍了主干网络的发展和检测算法的发展，在此省略。

FPN，detnet和传统方式的结构如下图所示：
![网络结构对比](http://upload-images.jianshu.io/upload_images/11609151-4f2a698526e6b8fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

FPN的结构带来了两个问题：
1.大尺度的物体在低分辨率特征图上检测，导致定位不准确。
2.尽管小尺度的物体可以在上采样的高分辨率特征图上进行检测，但是如果在下采样过程中这些小尺度的物体丢失，那么即便上采样，仍然不能很好的还原小尺度物体的语义信息。（个人认为这个理由有点牵强，毕竟网络在训练过程中如果考虑了小尺度的物体，那么训练完成后，网络应该有能力保证这个信息不丢失）

##detnet的设计

detnet的设计基于Resnet-50，在前4个stage保持和resnet一致。但是为了解决上述问题detnet的设计有两个挑战：
1. 增加的特征图分辨率会大大增加内存消耗
2. 减少了下采样会大大减小高层特征的感受野，这对于大多数计算机视觉任务是不好的。

基于resnet的修改从第五个stage开始。细节如下：
1. 增加一个stage6，并从stage5开始保持*16的感受野。
2. 在stage5和stage6采用了 dilated [29,30,31] bottleneck with 1x1 convolution projection（可能要去看论文，暂时不动怎么操作的）
3. 从stage4开始使用bottleneck with dilation 用于增加感受野，并且保持特征层维度为256不再改变，用于减少计算量。

具体的每一个stage的更改细节如下图所示：
![网络结构细节](http://upload-images.jianshu.io/upload_images/11609151-44e07a7f03c1ea22.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在测试时，选择的对照是FPN，将FPN使用的主干网络Resnet更换为detnet，其他的结构相同。（这里就显得创新性不够了，前面说了FPN的那么多问题，在比较的时候还是用了FPN的结构）

##实验
实验在COCO数据集下进行
部分实验结果如下:
![实验结果](https://upload-images.jianshu.io/upload_images/11609151-76ef35a51377c168.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![可视化结果](https://upload-images.jianshu.io/upload_images/11609151-541c0697a4ec9060.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





