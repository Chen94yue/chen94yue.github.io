个人之前没有tensorflow的使用经验，只接触过caffe，因此对于tensors的概念也是初步接触。如有谬误，欢迎指正。

官网的教程从tensor的使用入手。tensor的使用主要是为了能够在GPU上实现运算。算是数据和GPU的一个接口吧。该数据类型类似numpy，使用他需要加上以下预处理：
```python
form __future__ import print_function
import torch
```
初始化一个5*3的矩阵：
```python
x = torch.empty(5, 3)
print(x)
```
输出为：
```
tensor([[-7.7402e-04,  4.5621e-41, -2.4322e+20],
        [ 3.0967e-41,  1.8490e+20,  4.7468e+27],
        [ 5.5577e-11,  6.8571e+22,  2.5869e+20],
        [ 4.3991e+21,  1.8490e+20,  1.2120e+25],
        [ 6.7331e+22,  4.2326e+21,  1.6931e+22]])
```
可以看出是随机初始化的，每一次结果不同，但是和官网上的输出不太一样，不知道是否是pytorch的版本不同导致的，官网上该语句应该是生成未初始化的矩阵。

生成随机初始化的矩阵使用如下语句：
```
x = torch.rand(5, 3)
print(x)
```
结果如下：
```
tensor([[ 0.6325,  0.1186,  0.3710],
        [ 0.2410,  0.9606,  0.4492],
        [ 0.1594,  0.5641,  0.4554],
        [ 0.7680,  0.9980,  0.4455],
        [ 0.8256,  0.1882,  0.8410]])
```
构造一个全零的矩阵，并且用long型填充：
```
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```
结果如下：
```
tensor([[ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0]])
```
当然和其他矩阵操作一样，你可以通过：
```
x[1,1]
x[:,1]
```
这种指令读取其中一项或者一行，但是注意到使用type()函数并不能返回其中数据的类型，所有类型都返回的是：
```
<class 'torch.Tensor'>
```
其他的细节的使用方法可以参考官网：
```
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
```
该方法可以依据之前的张量生成新的张量，保持属性相同。
获得张量大小的语句为：
```
x.size()
```
两个张量相加, 调用函数或者直接相加：
```
torch.add(x, y)
x + y
y.add_(x)  #y +=x
```
改变张量的尺度：
```
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```
注意保持元素个数一样，输出分别是：
```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```
如果要获取一个位置的值将其作为python的数据类型，可以使用x.item()

更多的操作可以看[这里](https://pytorch.org/docs/stable/torch.html)

tensor提供可和numpy的接口，并且赋值之后是共享的：
1.改变tensor来改变numpy：
```
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)
输出为：
tensor([ 2.,  2.,  2.,  2.,  2.])
[2. 2. 2. 2. 2.]
```
2.改变numpy来改变tensor
```
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
输出为：
[2. 2. 2. 2. 2.]
tensor([ 2.,  2.,  2.,  2.,  2.], dtype=torch.float64)
```
在GPU上生成tensor：
```
if torch.cuda.is_available():
  device = torch.device("cuda")          # a CUDA device object
  y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
  x = x.to(device)                       # or just use strings ``.to("cuda")``
  z = x + y
  print(z)
  print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
输出为：
tensor([ 1.9422], device='cuda:0')
tensor([ 1.9422], dtype=torch.float64)
```


