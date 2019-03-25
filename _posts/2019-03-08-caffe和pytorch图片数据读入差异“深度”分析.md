---
layout:     
title:      Caffe和Pytorch图片数据读入差异“深度”分析
subtitle:   fuck you Caffe
date:       2019-03-08
author:     Shaozi
header-img: 
catalog: true
tags:
    - Caffe
    - Pytorch
---

问题背景：
	————近期工程上碰到这样一个情况，用pytorch训练的模型，在将网络参数转为caffemodel之后，在caffe下不能复现其性能，整体评价指标上差了百分之一点几。

为了判断到底是什么位置出现了问题。师兄首先做了控制变量的分析。去掉datalayer，直接送入一个自定义的矩阵给后面的网络。Pytorch和Caffe输出结果相同。因此将问题聚焦在caffe的datalayer上。接下来的工作交给我来做。

为了简化分析成本，我们只使用datalayer做以下几个简单的操作：
1. 读取图片
2. Resize图片
3. Normalize

我们使用如下图片（出自[CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)）进行测试：
![TestImage](https://i.loli.net/2019/03/08/5c822951d5640.jpg)

Pytorch用户通常使用的是torchvision下的transforms函数包对图像进行操作。该部分代码如下：
```python
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

img = pil_loader('American_Redstart_0064_103081.jpg')
trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
```
首先我们来看一下Resize的实现方法，源码为：
```python
def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
```
核心语句很简单，在对于输入的参数做了简单的判断和转换之后，调用了PIL库的resize函数，需要注意的是，这里使用的默认的差值方法为双线性差值（PIL.Image.BILINEAR）。看到这里就已经足够了，不用再去看PIL中resize是如何实现的，因为PIL好像没有完全开源？？？（我没有找到源码）

之后totensor的实现方法为：
```python
def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img
```
可以看到对于输入的PIL类型的图片，to_tensor函数首先依据其不同的图片结构进行转换，然后每一个像素除255将其归一化到[0,1],这也就是后面Normalize部分都是小数的原因。
由于Normalize为纯数字计算，前面的数值为每一个图像通道的均值，后面的数值为状态值。源码中的计算公式为：
```python
def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor
```
可以看到是减去均值之后再除状态值。

接下来我们来看caffe的实现，caffe这边首先定义网络的prototxt文件，因为我们不需要后面的网络，prototxt文件只包含一层网络：
```c++
layer {
  name: "Data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  image_data_param {
    source: "demo.txt"
    root_folder: "./"
    new_height: 224
    new_width: 224
    is_color: true
    batch_size: 1
  }
}
```
image data layer的代码就长了，这里就节选一下关键的部分放在这里(只看.cpp文件吧)：
```c++
cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,new_height, new_width, is_color);
CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
read_time += timer.MicroSeconds();
timer.Start();
// Apply transformations (mirror, crop...) to the image
int offset = batch->data_.offset(item_id);
this->transformed_data_.set_cpu_data(prefetch_data + offset);
this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
trans_time += timer.MicroSeconds();//统计预处理时间
```
首先调用ReadImageToCVMat函数读入图片，该函数在root/caffe/src/caffe/util/io.cpp中。
之后调用了data_transformer的Transform函数，该函数在root/caffe/src/caffe/data_transformer.cpp中。
首先ReadImageToCVMat的实现为：
```c++
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}
```
关键在于一句：cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
可以看到caffe的resize函数是基于opencv实现的。查一下opencv的文档，可以看到，该差值方法也是默认双线性差值。有兴趣的同学可以去[这里](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)查看一下。
另外resize的源码可以在[这里](https://github.com/opencv/opencv/blob/332c37f332733e5a2d717fc6eb4d605a304cda70/modules/imgproc/src/resize.cpp)看到。3764行，告辞🚓。

Caffe没有tensor这个东西，所以比Pytorch少了一步。最后看一下caffe的Normalize方法：
```c++
if (has_mean_file) {//若指定了均值文件
    transformed_data[top_index] =(datum_element - mean[data_index]) * scale;//执行去均值、幅度缩放
    } 
else {
    if (has_mean_values) {//若指定了均值数值
        transformed_data[top_index] =(datum_element - mean_values_[c]) * scale;//执行去均值、幅度缩放
    } 
    else {
     transformed_data[top_index] = datum_element * scale;//不去均值、只做幅度缩放
}
```
看到这里，第一个导致Caffe和Pytorch送入网络的数据不同的原因出现了————Normalize方法不同。
下面我们以Pytorch的为基准，来看一下Caffe的layer参数应该如何设置。推导过程很简单，我就省略了，这里设Caffe的参数scale为1/255为定值。mp，sp分别为Pytorch的一个通道的均值和状态值，mc为Caffe一个通道的均值。我们可以得到下面的关系：
```
根本推不出来！
```
但是我们还是将scale设为1吧。
新的datalayer为：
```c++
layer {
  name: "Data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: ???
    mean_value: ???
    mean_value: ???
    scale: 1/255
  }
  image_data_param {
    source: "demo.txt"
    root_folder: "./"
    new_height: 224
    new_width: 224
    is_color: true
    batch_size: 1
  }
}
```
没错这就是pytorch的第一个坑，我们根本没有办法找到完全对应的方法，那么有什么方法能够改变呢，其实很简单，修改Normalize函数的实现方式和caffe一样就行了。当然也可以修改caffe的实现方式和pytorch一样。难度不大。这里假设我们已经改为一样了，会有下面的网络定义：
```c++
layer {
  name: "Data"
  type: "ImageDataPytorch"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 0.485
    mean_value: 0.456
    mean_value: 0.406
    std_value: 0.229
    std_value: 0.224
    std_value: 0.225
  }
  image_data_param {
    source: "demo.txt"
    root_folder: "./"
    new_height: 224
    new_width: 224
    is_color: true
    batch_size: 1
  }
}
```
源码我懒得改了，大致就是(pixel_value/225 - mean_value)/std_value。保持和Pytorch一样就行了。

经过上面的修改应该两个输出一样了吧？没错，还不一样。
那么问题就出在resize函数上了。
如果直接输出，你会发现，我的天，差距好大啊！！！
都是双线性差值，能差距这么大？
其实这里第一个不同是PIL的图像矩阵为W\*H\*C，而Opencv的是H\*W\*C，不过这一点并不影响网络的计算，因为在后面网络计算中已经考虑这一点区别了，在Pytorch的ToTensor操作中已经考虑了这一点，在对Numpy类型的数据进行转换时做了相应的矩阵变换操作。真正存在问题的是PIL读入的图片是RGB排列的三个通道，而Opencv读入的是BGR排列的三个通道。这里需要我们在ReadImageToCVMat函数的输出前加上一行，可破此阵。
```c++
cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);
```
可是即便这样，RGB三通道的输出像素差别也较大。下面画了三张像素差值的分布图：
R：
![R](https://i.loli.net/2019/03/08/5c82423f1f783.png)
G：
![G](https://i.loli.net/2019/03/08/5c82423f336af.png)
B：
![B](https://i.loli.net/2019/03/08/5c82423eab405.png)
最大的偏差甚至大于70个像素。

这样我们可能要正面面对三千多行的opencv源码了。但是PIL如何实现的我们并不能看到。
还好，Pytorch存在一定的人性，Opencv也有python的库，cv2。我们只需要在Pytorch训练时，不使用torchvision提供的工具改变图片，避开PIL库即可。
将Pytorch的代码改为：
```python
img = cv2.imread('American_Redstart_0064_103081.jpg')
img = cv2.resize(img, (224,224))
img_t = transforms.Compose([transforms.ToTensor()])(img)
```
自此可以保证送入网络的数据完全一样！








