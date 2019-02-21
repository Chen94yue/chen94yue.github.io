18年CVPR收录的论文目录点[这里](http://cvpr2018.thecvf.com/program/main_conference)

粗略估计，今年做检测任务的共有103篇，跟行人有关的研究有50篇，其中大量研究的任务是行人重识别，研究道路行人的有五篇，主要针对的是拥挤场景，本文为其中之一。[本文链接](https://arxiv.org/abs/1711.07752)

论文来自同济大学和清华大学，其中第二作者Tete Xiao的另外一篇做行人检测的文章

>What Can Help Pedestrian Detection?

出自2017年的CVPR，同样值得研究。

本文针对行人检测中的遮挡问题，提出了一种排斥损失，用来约束检测器的推荐区域。

行人的遮挡主要来自两种情况，一种是其他类别物体的遮挡，另一种是个体之间的遮挡。而在研究中红，个体之间的遮挡占主要的情况。因为在智能监控和道路行人检测领域，行人往往习惯于以群体的形式出现。

![图 1](http://upload-images.jianshu.io/upload_images/11609151-049a26bd25ade976.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图展示的是个体之间遮挡对算法检测效果的影响。遮挡的存在不仅使得算法在定位上不准确，也使得算法对NMS的阈值的要求提高。

因此本文考虑将周围个体之间的影响也纳入到损失的计算中来。简单的说就是使预测框靠近检测目标的GT（ground truth）。远离其他目标和替他GT。

*文章首先研究了遮挡出现的情况，这里论文中有详细的介绍，就不再细说了。这一部分作为文章的第一个贡献出现。其实也给之后的论文写作提供了一种“凑字数”的途径。*

下面详细介绍一下本文提出的loss。

loss的结构如下：
![image](http://upload-images.jianshu.io/upload_images/11609151-c86a1ef09d7354d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

各部分由公式显然可得使什么意思。为了介绍的简单，后面只考虑单类别的任务：设P和G是检测到的结果和GT，*P+*是所有检测到的正样本的集合，*G*是所有GT的集合。

其中吸引力部分的损失如下：B是回归得到的bbox
![image](http://upload-images.jianshu.io/upload_images/11609151-94b90189f5f6331d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和GT的相斥损失定义为：
![image](http://upload-images.jianshu.io/upload_images/11609151-2a3165f4d352c5cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
实际上就是找到和检测结果重合最大的非目标GT然后尽量减少这个重合。其中*Gp_Rep*就是指i这个GT。需要说的是这里的IoG不同于IoU，IoG是对与GT算的重叠部分的比例。

和其他检测结果的相斥损失，首先按照检测结果对应的GT将其进行划分，然后不同分组中的检测结果之间的距离要尽量的大：
![image](http://upload-images.jianshu.io/upload_images/11609151-c25bf0b4e02d20de.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

*个人认为这个loss在严格意义上不会提高检测的准确度，但是确实降低了筛选检测结果时NMS的阈值对检测效果影响。在计算相斥的损失的时候使用IOU而不使用smooth是因为损失只需要求最小即可，并不需要尽量的小*

实验验证了上面的参数设置和各部分的效果，在此略去。就整体和其他算法的对比实验看，效果是最好的。
