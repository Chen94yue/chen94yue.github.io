#functional
（\*为较为常用的函数）

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


