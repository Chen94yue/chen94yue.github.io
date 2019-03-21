---
layout:     post
title:      Torchvision扩充：基于Opencv的图像增强
subtitle:   Pytorch
date:       2019-03-22
author:     Shaozi
header-img: 
catalog: true
tags:
    - Pytorch
    - Opencv
---

原始的torchvision工具包只支持PIL图像的各种增强操作，模型转换为caffe后，输入图像存在偏差，导致模型性能下降。为了消除该问题，需要实现一套基于opencv的图像增强工具，本文在原先torchvision.transform的基础上进行修改，主要涉及两个文件：functional.py和transforms.py，测试通过的工具包会在文末的地址开源。

###functional

（\*为较为常用的函数）

目前原始输入只支持输入为RGB彩色图像和灰度图像

0. _is_pil_image

原始：判断图片是否为PIL格式的数据

修改：删除

0. _is_tensor_image

原始：判断是否图像类型为tensor

修改：不变

0. _is_numpy_image

原始：判断图像是否为numpy，由于opencv读入之后默认为numpy，故该函数用于判断是否为图像

修改：不变

0. to_tensor

原始：支持PIL和numpy类型的图像

修改：去掉对PIL图像的支持

0. to_pil_image

原始：将PIL或numpy转换为PIL图片

修改：现在没有这个需求了，删除该函数

0. normalize\*

原始：tensor均一化

修改：pytorch的均一化方法和caffe不同，两条路，保持原有方法，修改caffe源码，或修改此函数，现将该函数保留另增加函数normalize_caffe

0. normalize_caffe\*

说明:按照caffe的均一化方式计算，需提供scale(默认为1)和mean_value

0. resize\*

原始：基于PIL实现

修改：基于opencv实现,注意逻辑保持和pytorch一样，若只指定一个resize参数，保证的是短边和该数一样，长边做等比例缩放，指定两个参数（h,w），则严格按照该参数进行

0. scale

原始：等价于resize

修改：不变

0. pad\*

原始：按指定的方式填充图片边缘支持RGB和灰度

修改：由于CV2读取灰度图会自动填充为三通道，故删去对单通道图片的支持。

0. crop\*

原始：基于PIL实现

修改：基于cv2实现

0. center_crop\*

同上

0. resized_crop\*

同上

0. hflip\*

同上

0. vflip\*

同上

0. five_crop\*

同上

0. ten_crop\*

原始：基于five_crop和flip实现

修改：不变

0. adjust_brightness

原始：基于PIL的ImageEnhance工具库实现，源码不可见。输入为图像和亮度变换的比例（0，+∞）（等比例相乘）

修改：实现方式不同，只能说达到了相同的功能，输入为图像和亮度变换的数值（-∞，+∞）（数值相加，当像素范围超过[0,255]时设为0或255）

0. adjust_contrast

原始：基于PIL的ImageEnhance工具库实现，源码不可见。输入为图像和对比度变换的比例（0，+∞）

修改：a \* image,用于修改对比度。

0. adjust_saturation

原始：同上

修改：基于opencv实现，由于没有现成的函数，只能自己设计算法，考虑到该函数不常用，使用了一个较为简单的手法，将BGR的图像转为HSV，然后修改S通道的值，之后再转回BGR

0. adjust_hue

同上

0. adjust_gamma

原始：对图像进行伽马矫正，在图像增强中很少用

修改：转为CV2实现

0. affine

原始：图像的仿射变换，基于PIL的工具包实现

修改：虽然opencv也有仿射变换的实现方法，要实现此功能是可行的，但是变换原理和现有的原理差别较大，基于opencv复现后不能保证和PIL的实现方式等价。考虑该函数使用较少，先放弃修改该函数。若之后有需求再做深究

0. to_grayscale

原始：将原始的RGB图像转为灰度图，并且可选单通道或相同的三通道输出。

修改：考虑opencv的灰度图本来就是相同的三个通道，故不修改该函数。

###tranforms
