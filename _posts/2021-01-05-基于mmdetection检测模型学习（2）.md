---
layout:     post
title:      基于mmdetection检测模型学习（2）：关于NMS
subtitle:   mmdetection
date:       2021-01-08
author:     Shaozi
header-img:
catalog: true
tags:
- mmdetection
- detection
---

## 总览

在Faster-RCNN的配置文件中，关于nms的设置放在[model training and testing settings](https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/models/faster_rcnn_r50_fpn.py)下，分别作用于rpn和rcnn部分网络的输出，在rpn部分设置的参数包括：

```python
nms_across_levels=False,
nms_pre=2000,
nms_post=1000,
nms_thr=0.7,
```

在rcnn部分设置的参数包括：

```python
score_thr=0.05,
nms=dict(type='nms', iou_threshold=0.5),
max_per_img=100
```

注意到在train_cfg的rcnn部分未设置相关参数。

## RPN的nms实现

- nms_across_levels

    该参数只在`GARPNHead`下用到，由于Faster-rcnn中设置的普通的`RPNHead`所以该参数并不会真实的使用。该参数会作用在RPN生成proposal的时候，若设为True，则会对rpn网络生成的所有level的proposal一起做nms，采用的nms方式为mmcv.ops下的[nms](https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/nms.py)函数。具体的，这里采用了最基础的nms，相应的nms_thr设置的就是此时iou的阈值，具体的nms实现方法见[nms.cpp](https://github.com/open-mmlab/mmcv/blob/005c408748446bab1615873c46ae65a3a7124a29/mmcv/ops/csrc/pytorch/nms.cpp)下的nms函数。对应的nms_thr设置的越小，proposal之间的重叠程度越低，proposal数量越小。

- nms_pre

    如果一个level生产的proposal数量大于nms_pre，则会按照score进行排序之后取前nms_pre个保留。

- nms_thr

    NMS的iou阈值。**nms_thr设置的越小，proposal之间的重叠程度越低，proposal数量越小**。

- nms_post

    最后留下的所有level的nms结果数量的最大值。**真正nms结果的输出**。

*值得注意的是，rpn中nms的实现对于每一个level是相互独立的，代码将level编号设计成了类别的label，之后调用`batch_nms`巧妙的实现了这一点。关于`batch_nms`,会在本文之后进行分析。*

## RCNN的nms实现

找到RCNN的nms实现还是比较复杂的，这里需要一层一层的看。

首先Faster-rcnn使用的roi_head为[StandardRoIHead](https://github.com/open-mmlab/mmdetection/blob/4e921b2f40edd461f9219c1cf831b8b85b3f569a/mmdet/models/roi_heads/standard_roi_head.py),该类有三个父类，分别是BaseRoIHead, BBoxTestMixin, MaskTestMixin。直接看该类的`forward_train`函数，在其中调用了`_bbox_forward_train`来获得bbox结果，在`_bbox_forward_train`中又通过`_bbox_forward`来从输入的特征和roi区域得到bbox的预测结果。之后会使用`self.bbox_head`的`get_targets`获得bbox的结果。

Faster-rcnn的bbox_head使用了[Shared2FCBBoxHead](https://github.com/open-mmlab/mmdetection/blob/4fb980e73e283ee55a6a8591e1e2057ed98cd1fa/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py)，因为该类继承了`ConvFCBBoxHead`，并且只是修改了初始化的参数值，所以直接看`ConvFCBBoxHead`的下的`get_targets`实现，发现`ConvFCBBoxHead`类也没有该函数的实现，所以再去他的父类去看，在[BBoxHead](https://github.com/open-mmlab/mmdetection/blob/4fb980e73e283ee55a6a8591e1e2057ed98cd1fa/mmdet/models/roi_heads/bbox_heads/bbox_head.py#L13)类中，`get_targets`的实现实际是多次调用了`_get_target_single`，在`_get_target_single`中我们其实可以看到实际没有使用任何nms的操作。所以可以认为训练时RCNN部分并没有做nms的操作。

之后我们来考察test时的实现，这里回到[StandardRoIHead](https://github.com/open-mmlab/mmdetection/blob/4e921b2f40edd461f9219c1cf831b8b85b3f569a/mmdet/models/roi_heads/standard_roi_head.py)，我们观察`simple_test`函数（`aug_test`函数应该只是增加了一个针对图像增强的图像映射转换）。这里调用了`simple_test_bboxes`找到这个函数，需要去他的父类中找，我们可以从[BBoxTestMixin](https://github.com/open-mmlab/mmdetection/blob/4fb980e73e283ee55a6a8591e1e2057ed98cd1fa/mmdet/models/roi_heads/test_mixins.py)中找到这个函数。这个函数实际调用的是`Shared2FCBBoxHead`的`get_bboxes`函数来获取bbox的，该函数的实现还是位于[BBoxHead](https://github.com/open-mmlab/mmdetection/blob/4fb980e73e283ee55a6a8591e1e2057ed98cd1fa/mmdet/models/roi_heads/bbox_heads/bbox_head.py#L13)。最终可以定位Faster-rcnn的的nms只在测试时使用，并且通过[multiclass_nms](https://github.com/open-mmlab/mmdetection/blob/4fb980e73e283ee55a6a8591e1e2057ed98cd1fa/mmdet/core/post_processing/bbox_nms.py#L7)函数实现，通过该函数的实现，可以看到config中各项参数设置的含义分别为：

- score_thr

    获得的bbox的最低得分值，分数低于该阈值的bbox会在nms之前被剔除。

- max_per_img

    网络输出的最终结果中的bbox的数量

- nms

    调用`batch_nms`时的设置。

## batch_nms

batch_nms的实现实际上是在基本的单类别的nms的基础上，增加了按照不同类别进行nms的功能。代码如下：

```python
def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    # 判断是否要分类别做nms
    if class_agnostic:
        # 不分类别，则所有box一起算
        boxes_for_nms = boxes
    else:
        # 分类别
        # 找到box的最大坐标值
        max_coordinate = boxes.max()
        # 按照label对box做一个偏置，偏置的值等于label*最大坐标。
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        # bbox坐标加上偏置，相当于将不同类别的bbox平移到了不同的平面区域
        boxes_for_nms = boxes + offsets[:, None]

    # 获取nms的类型
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    
    # 对bbox进行拆分，之后分组做nms
    split_thr = nms_cfg_.pop('split_thr', 10000)
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep
```



