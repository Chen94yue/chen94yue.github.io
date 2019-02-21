虽然我在前面写到没有必要去阅读整个官方文档，但是在开发过程中发现，如果对整个文档特别是关于tensor的操作和函数有一定的了解，那么实际运用起来是事半功倍的。只能说一句“真香”来概括我现在的心情了。所以后面还是会详细的把官方文档中所有的函数做一个学习和总结。这一部分工程量很大，不知道啥时候可以完成。。。。

#Tensors
##属性相关
```python
torch.is_tensor(obj)
```
用来判断obj是否为一个tensor变量。同样的还有torch.is_storage(obj)用于判断变量是否为一个storage变量。
```python
torch.set_default_dtype(d)
```
用于改变tensor中数据的默认类型，如下面的例子：
```python
>>> torch.tensor([1.2, 3]).dtype           # initial default for floating point is torch.float32
torch.float32
>>> torch.set_default_dtype(torch.float64)
>>> torch.tensor([1.2, 3]).dtype           # a new floating point tensor
torch.float64
```
所有支持的数据类型如下表所示：
![tensor数据类型](https://upload-images.jianshu.io/upload_images/11609151-5b3528de1242a5e2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
想要得到tensor中默认的数据类型，可以使用：torch.get_default_dtype()，如下面的这个例子：
```python
>>> torch.get_default_dtype()  # initial default for floating point is torch.float32
torch.float32
>>> torch.set_default_dtype(torch.float64)
>>> torch.get_default_dtype()  # default is now changed to torch.float64
torch.float64
>>> torch.set_default_tensor_type(torch.FloatTensor)  # setting tensor type also affects this
>>> torch.get_default_dtype()  # changed to torch.float32, the dtype for torch.FloatTensor
torch.float32
```
同时也提供了设置默认的tensor的数据类型的函数，但是我没看懂这个和前面的做法有什么区别：
```
torch.set_default_tensor_type(t)
```
例子如下：
```
>>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
torch.float32
>>> torch.set_default_tensor_type(torch.DoubleTensor)
>>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
torch.float64
```
为了得到tensor中所有的元素个数可以使用：
```
torch.numel(input)
```
函数返回一个int型的数据，例子如下：
```
>>> a = torch.randn(1, 2, 3, 4, 5)
>>> torch.numel(a)
120
>>> a = torch.zeros(4,4)
>>> torch.numel(a)
16
```
当你需要输出tensor查看的时候，或许需要设置一下默认的输出选项：
```
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)
```
其中precision是每一个元素的输出精度，默认是八位；threshold是输出时的阈值，当tensor中元素的个数大于该值时，进行缩略输出，默认时1000；edgeitems是输出的维度，默认是3；linewidth字面意思，每一行输出的长度；profile=None，修正默认设置（不太懂，感兴趣的可以试试）)

为了防止一些不正常的元素产生，比如特别小的数，pytorch支持如下设置：
```
torch.set_flush_denormal(mode)
```
mode中可以填true或者false
例子如下：
```
>>> torch.set_flush_denormal(True)
True
>>> torch.tensor([1e-323], dtype=torch.float64)
tensor([ 0.], dtype=torch.float64)
>>> torch.set_flush_denormal(False)
True
>>> torch.tensor([1e-323], dtype=torch.float64)
tensor(9.88131e-324 *
       [ 1.0000], dtype=torch.float64)
```
可以看出设置了之后，当出现极小数时，直接置为0了。文档中提出该功能必须要系统支持。

##Creation Ops
这一部分主要介绍tensor的生成。首先直接赋值：
```
torch.tensor(data, dtype=None, device=None, requires_grad=False)
```
具体的结果如下例子所示：
```
>>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor([[ 0.1000,  1.2000],
        [ 2.2000,  3.1000],
        [ 4.9000,  5.2000]])

>>> torch.tensor([0, 1])  # Type inference on data
tensor([ 0,  1])

>>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
                 dtype=torch.float64,
                 device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

>>> torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
tensor(3.1416)

>>> torch.tensor([])  # Create an empty tensor (of size (0,))
tensor([])
```
当然不能忘了tensor和numpy之间可以相互转换，这个在前面基础部分就有介绍，使用torch.from_numpy即可。
```
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1,  2,  3])
>>> t[0] = -1
>>> a
array([-1,  2,  3])
```
如果要生成全为0的tensor，我们使用torch.zeros()，参数如下：
```
torch.zeros(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```
其中size可以是任意维度的list或tuple。如下所示：
```
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

>>> torch.zeros(5)
tensor([ 0.,  0.,  0.,  0.,  0.])
```
另外还有一个zeros_like函数，从函数名不难猜到，这个函数是用于生成和输入tensor大小相同的全零tensor的。
处理生成全零的tensor，还有one()函数，用于生成全为1的tensor。也有one_like函数。
下面这个函数和python中的range类似，用于产生一个一维的tensor，在给定的区间下依据给定的步长。
```
torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```
例子如下所示：
```
>>> torch.arange(5)
tensor([ 0.,  1.,  2.,  3.,  4.])
>>> torch.arange(1, 4)
tensor([ 1.,  2.,  3.])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```
注意这里生成的tensor中是不包含上界的，如果要包含上界，可以使用range替代。当然也可以通过设定上下界和元素的个数来避免步长的设定。使用linspace就行了：
```
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
>>> torch.linspace(start=-10, end=10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])
```
既然有线性空间，那么log空间自然也是支持的。（虽然我暂时不知道这个可以用来干嘛？画图吗？）
将上面的linspace换为logspace即可：
```
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
>>> torch.logspace(start=0.1, end=1.0, steps=5)
tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
```
单位矩阵在矩阵运算中起到了很关键的作用，需要生成一个单位阵，可以使用如下语句：
```
>>> torch.eye(3)
tensor([[ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]])
```
当然如果特殊需要，单位阵也是支持设置宽的：
```
>>> torch.eye(3,7)
tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.]])
```
除了单位阵，还可以生成未初始化的矩阵，调用empty即可，数字是随机的：
```
>>> torch.empty(2, 3)
tensor(1.00000e-08 *
       [[ 6.3984,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000]])
```
该函数同样有兄弟：empty_like()
前面介绍了可以生成全为0和1的tensor，那么如果我要生成全为2的呢？
首先你可以拿全为1的乘以一个常数，其次，你可以使用full()函数：
```
>>> torch.full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```
同样full函数有基友full_like()。
以上是pytorch目前支持的所有tensor生成方法，下面介绍关于tensor的一系列“矩阵操作”。
##Indexing, Slicing, Joining, Mutating Ops
这一部分将介绍目前pytorch支持的所有关于tensor的各种变换操作。首先是多个tensor的连接，这里和caffe的concat layer作用应该类似，但是不得不感叹，pytorch的实现简洁太多了啊。函数如下：
```
torch.cat(seq, dim=0, out=None)
```
例子如下，做的事二维的，当然高维的同样也可以，不过你必须保证连接的维度上的长度事匹配的。
```
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```
通过设置参数dim来选择按哪一个维度相连。
还有另外一种连接的方法：
```
torch.stack(seq, dim=0, out=None) 
```
该函数跟cat基本相同。不过stack要求连接的tensor每一个维度都相同。

同样可以将一个tensor分割开来，函数如下：
```
torch.chunk(tensor, chunks, dim=0) 
```
例子如下：
```
>>> print(a)
tensor([[ 0.6253,  0.8666, -0.1230,  0.3984,  0.2968],
        [-1.1441,  1.1067, -0.0283,  0.4503, -0.4435],
        [ 0.4108, -0.2321,  0.2295,  0.2917,  0.1316],
        [ 1.4066, -0.2489,  0.2258, -0.5783, -0.6589],
        [-1.9384,  0.8134,  0.2353, -0.1845, -1.1675],
        [-0.7617,  0.6622,  0.6844,  0.0229, -0.7072],
        [ 0.7110, -0.8292, -0.1205,  1.3795, -1.3677],
        [-0.0562,  1.6998, -0.2817, -0.7298,  0.2130],
        [ 0.4300,  0.8207, -1.1832,  0.9723, -0.0193],
        [-0.3227,  0.1291, -0.1117, -0.2469, -0.5320]])
>>> torch.chunk(a,2,0)
(tensor([[ 0.6253,  0.8666, -0.1230,  0.3984,  0.2968],
        [-1.1441,  1.1067, -0.0283,  0.4503, -0.4435],
        [ 0.4108, -0.2321,  0.2295,  0.2917,  0.1316],
        [ 1.4066, -0.2489,  0.2258, -0.5783, -0.6589],
        [-1.9384,  0.8134,  0.2353, -0.1845, -1.1675]]), tensor([[-0.7617,  0.6622,  0.6844,  0.0229, -0.7072],
        [ 0.7110, -0.8292, -0.1205,  1.3795, -1.3677],
        [-0.0562,  1.6998, -0.2817, -0.7298,  0.2130],
        [ 0.4300,  0.8207, -1.1832,  0.9723, -0.0193],
        [-0.3227,  0.1291, -0.1117, -0.2469, -0.5320]]))
>>> torch.chunk(a,2,1)
(tensor([[ 0.6253,  0.8666, -0.1230],
        [-1.1441,  1.1067, -0.0283],
        [ 0.4108, -0.2321,  0.2295],
        [ 1.4066, -0.2489,  0.2258],
        [-1.9384,  0.8134,  0.2353],
        [-0.7617,  0.6622,  0.6844],
        [ 0.7110, -0.8292, -0.1205],
        [-0.0562,  1.6998, -0.2817],
        [ 0.4300,  0.8207, -1.1832],
        [-0.3227,  0.1291, -0.1117]]), tensor([[ 0.3984,  0.2968],
        [ 0.4503, -0.4435],
        [ 0.2917,  0.1316],
        [-0.5783, -0.6589],
        [-0.1845, -1.1675],
        [ 0.0229, -0.7072],
        [ 1.3795, -1.3677],
        [-0.7298,  0.2130],
        [ 0.9723, -0.0193],
        [-0.2469, -0.5320]]))
```
返回的值是一个tuple。将其依次复制给需要的变量即可。
同样还有另外一个划分的方法，重写的split函数，功能相似：
```
torch.split(tensor, split_size_or_sections, dim=0)
```
其中split_size_or_sections用于设定划分规则，dim用于设定划分的维度。如下例：
```
>>> x = torch.randn(5, 5, 5)
>>> x
tensor([[[-0.8347,  2.2014,  0.2768,  0.8642,  0.7517],
         [-1.0237,  1.1800, -1.9238,  0.7537, -0.7155],
         [-0.5800, -0.0527,  1.1536,  0.8828,  0.0136],
         [ 0.1452, -0.5878, -1.9840,  0.8288,  1.1804],
         [-0.0514,  0.6633, -1.0233, -1.6492,  0.2422]],

        [[-0.1809,  0.0115, -0.1156,  0.4199, -0.6424],
         [ 1.0098,  0.2440,  0.2432,  1.5793, -0.4407],
         [-0.5316,  1.6012, -0.4609,  2.3206,  1.1053],
         [ 0.0999, -0.3938, -1.4438,  0.4003, -0.6948],
         [-1.5991, -0.2188,  0.8460,  1.2603, -0.4484]],

        [[ 0.1010,  1.8040,  0.3617,  0.5746, -0.0840],
         [-0.0948, -0.7579,  1.4936, -1.1720, -0.0606],
         [ 0.9900, -0.8400,  0.5351, -0.5388, -1.7400],
         [-0.5397, -0.1141, -0.3179, -0.2871, -0.8846],
         [ 0.3530,  0.5373,  0.2718,  1.2432, -1.3468]],

        [[ 0.5830,  0.4154,  0.9897,  1.6300,  1.5950],
         [ 0.4810, -1.1237, -0.2133,  0.5659, -0.4047],
         [ 0.0257, -0.3569,  1.4560, -0.8150, -0.3167],
         [ 1.8086,  0.1829, -0.7073, -1.1966,  0.3805],
         [ 0.0532, -0.2976,  0.3080, -1.3165, -1.0769]],

        [[ 1.8177, -0.6848,  0.1681, -1.0104,  0.9661],
         [-1.1784, -1.7252,  0.5589, -1.5597,  1.1723],
         [ 0.5254, -1.3278,  2.0289, -1.6005, -0.9900],
         [-0.2772, -0.4890,  1.1362,  0.9137,  1.0255],
         [-0.3162, -0.6244,  0.9933, -1.7472,  0.5968]]])
>>> torch.split(x,2,0)
(tensor([[[-0.8347,  2.2014,  0.2768,  0.8642,  0.7517],
         [-1.0237,  1.1800, -1.9238,  0.7537, -0.7155],
         [-0.5800, -0.0527,  1.1536,  0.8828,  0.0136],
         [ 0.1452, -0.5878, -1.9840,  0.8288,  1.1804],
         [-0.0514,  0.6633, -1.0233, -1.6492,  0.2422]],

        [[-0.1809,  0.0115, -0.1156,  0.4199, -0.6424],
         [ 1.0098,  0.2440,  0.2432,  1.5793, -0.4407],
         [-0.5316,  1.6012, -0.4609,  2.3206,  1.1053],
         [ 0.0999, -0.3938, -1.4438,  0.4003, -0.6948],
         [-1.5991, -0.2188,  0.8460,  1.2603, -0.4484]]]), tensor([[[ 0.1010,  1.8040,  0.3617,  0.5746, -0.0840],
         [-0.0948, -0.7579,  1.4936, -1.1720, -0.0606],
         [ 0.9900, -0.8400,  0.5351, -0.5388, -1.7400],
         [-0.5397, -0.1141, -0.3179, -0.2871, -0.8846],
         [ 0.3530,  0.5373,  0.2718,  1.2432, -1.3468]],

        [[ 0.5830,  0.4154,  0.9897,  1.6300,  1.5950],
         [ 0.4810, -1.1237, -0.2133,  0.5659, -0.4047],
         [ 0.0257, -0.3569,  1.4560, -0.8150, -0.3167],
         [ 1.8086,  0.1829, -0.7073, -1.1966,  0.3805],
         [ 0.0532, -0.2976,  0.3080, -1.3165, -1.0769]]]), tensor([[[ 1.8177, -0.6848,  0.1681, -1.0104,  0.9661],
         [-1.1784, -1.7252,  0.5589, -1.5597,  1.1723],
         [ 0.5254, -1.3278,  2.0289, -1.6005, -0.9900],
         [-0.2772, -0.4890,  1.1362,  0.9137,  1.0255],
         [-0.3162, -0.6244,  0.9933, -1.7472,  0.5968]]]))
>>> torch.split(x,(3,2),0)
(tensor([[[-0.8347,  2.2014,  0.2768,  0.8642,  0.7517],
         [-1.0237,  1.1800, -1.9238,  0.7537, -0.7155],
         [-0.5800, -0.0527,  1.1536,  0.8828,  0.0136],
         [ 0.1452, -0.5878, -1.9840,  0.8288,  1.1804],
         [-0.0514,  0.6633, -1.0233, -1.6492,  0.2422]],

        [[-0.1809,  0.0115, -0.1156,  0.4199, -0.6424],
         [ 1.0098,  0.2440,  0.2432,  1.5793, -0.4407],
         [-0.5316,  1.6012, -0.4609,  2.3206,  1.1053],
         [ 0.0999, -0.3938, -1.4438,  0.4003, -0.6948],
         [-1.5991, -0.2188,  0.8460,  1.2603, -0.4484]],

        [[ 0.1010,  1.8040,  0.3617,  0.5746, -0.0840],
         [-0.0948, -0.7579,  1.4936, -1.1720, -0.0606],
         [ 0.9900, -0.8400,  0.5351, -0.5388, -1.7400],
         [-0.5397, -0.1141, -0.3179, -0.2871, -0.8846],
         [ 0.3530,  0.5373,  0.2718,  1.2432, -1.3468]]]), tensor([[[ 0.5830,  0.4154,  0.9897,  1.6300,  1.5950],
         [ 0.4810, -1.1237, -0.2133,  0.5659, -0.4047],
         [ 0.0257, -0.3569,  1.4560, -0.8150, -0.3167],
         [ 1.8086,  0.1829, -0.7073, -1.1966,  0.3805],
         [ 0.0532, -0.2976,  0.3080, -1.3165, -1.0769]],

        [[ 1.8177, -0.6848,  0.1681, -1.0104,  0.9661],
         [-1.1784, -1.7252,  0.5589, -1.5597,  1.1723],
         [ 0.5254, -1.3278,  2.0289, -1.6005, -0.9900],
         [-0.2772, -0.4890,  1.1362,  0.9137,  1.0255],
         [-0.3162, -0.6244,  0.9933, -1.7472,  0.5968]]]))
>>> torch.split(x,(3,2),1)
(tensor([[[-0.8347,  2.2014,  0.2768,  0.8642,  0.7517],
         [-1.0237,  1.1800, -1.9238,  0.7537, -0.7155],
         [-0.5800, -0.0527,  1.1536,  0.8828,  0.0136]],

        [[-0.1809,  0.0115, -0.1156,  0.4199, -0.6424],
         [ 1.0098,  0.2440,  0.2432,  1.5793, -0.4407],
         [-0.5316,  1.6012, -0.4609,  2.3206,  1.1053]],

        [[ 0.1010,  1.8040,  0.3617,  0.5746, -0.0840],
         [-0.0948, -0.7579,  1.4936, -1.1720, -0.0606],
         [ 0.9900, -0.8400,  0.5351, -0.5388, -1.7400]],

        [[ 0.5830,  0.4154,  0.9897,  1.6300,  1.5950],
         [ 0.4810, -1.1237, -0.2133,  0.5659, -0.4047],
         [ 0.0257, -0.3569,  1.4560, -0.8150, -0.3167]],

        [[ 1.8177, -0.6848,  0.1681, -1.0104,  0.9661],
         [-1.1784, -1.7252,  0.5589, -1.5597,  1.1723],
         [ 0.5254, -1.3278,  2.0289, -1.6005, -0.9900]]]), tensor([[[ 0.1452, -0.5878, -1.9840,  0.8288,  1.1804],
         [-0.0514,  0.6633, -1.0233, -1.6492,  0.2422]],

        [[ 0.0999, -0.3938, -1.4438,  0.4003, -0.6948],
         [-1.5991, -0.2188,  0.8460,  1.2603, -0.4484]],

        [[-0.5397, -0.1141, -0.3179, -0.2871, -0.8846],
         [ 0.3530,  0.5373,  0.2718,  1.2432, -1.3468]],

        [[ 1.8086,  0.1829, -0.7073, -1.1966,  0.3805],
         [ 0.0532, -0.2976,  0.3080, -1.3165, -1.0769]],

        [[-0.2772, -0.4890,  1.1362,  0.9137,  1.0255],
         [-0.3162, -0.6244,  0.9933, -1.7472,  0.5968]]]))
```

下面这个函数应该是到目前为止，第一个不太好理解的。
```
torch.gather(input, dim, index, out=None) 
```
直接上例子吧，详细的解释和应用可以去这个[博客](https://blog.csdn.net/IAMoldpan/article/details/78660068)，因为我确实没有用到过，所以只能从算法上简单的解释一下了。
这个函数的意义在于按照一个给定的轴收集原tensor中的值，并得到一个新的tensor，其中dim = 0 是按照y轴，dim = 1是按照x轴，按照哪个轴在原tensor和index对应的tensor中都是按照相同的轴读取。输出也按照该轴输出。这一点是没法一眼看懂这个函数的原因所在。例子如下：
```
>>> t = torch.tensor([[1,2],[3,4]])
>>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
tensor([[ 1,  1],
        [ 4,  3]])
>>> torch.gather(t, 0, torch.tensor([[0,0],[1,0]]))
tensor([[ 1,  2],
        [ 3,  2]])
```
当dim为1时，按照x轴（行）读取，index对应的第一行为0，0所以连续在原tensor中读取两次第一个位置，保存在结果tenser的第一行。第二行同理。
当dim为0时，按照y轴（列）读取，index对应的第一列为0，1所以分别读取原tensor的第一列的第一个第二个数1和3存在结果tensor的第一列。

和上面挑tensor中元素的情况类似，我们还能挑选tensor中的一整行和一整列。函数如下：
```
torch.index_select(input, dim, index, out=None)
```
这个例子还是很好理解的，我觉得我不需要过多的解释了。直接上结果：
```
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-0.4664,  0.2647, -0.1228, -1.1068],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> indices = torch.tensor([0, 2])
>>> torch.index_select(x, 0, indices)
tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
        [-1.1734, -0.6571,  0.7230, -0.6004]])
>>> torch.index_select(x, 1, indices)
tensor([[ 0.1427, -0.5414],
        [-0.4664, -0.1228],
        [-1.1734,  0.7230]])
```
如果想要挑选出tensor中所有大于一个阈值的量要怎么做呢？
这里要结合两个函数，ge()和mask_select()这里首先介绍mask_select()函数，ge函数在后面的tensor运算章节会讲。这里只需要知道他讲输入tensor按照一个阈值01化即可（大于阈值设为1）。mask_select函数将按照输入的mask挑选出原tensor中mask上面为1的元素，生成一个一维的向量。
结果如下：
```
>>> x = torch.randn(3, 4)
>>> x
tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
        [-1.2035,  1.2252,  0.5002,  0.6248],
        [ 0.1307, -2.0608,  0.1244,  2.0139]])
>>> mask = x.ge(0.5)
>>> mask
tensor([[ 0,  0,  0,  0],
        [ 0,  1,  1,  1],
        [ 0,  0,  0,  1]], dtype=torch.uint8)
>>> torch.masked_select(x, mask)
tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
```
下面这个函数用于得到tensor中非零元素的位置坐标，返回值的每一行代表一个坐标。如果输入tensor是n维的，其中非零元素个数为k，那么返回值是一个k×n的tensor。函数和示例如下所示：
```
torch.nonzero(input, out=None) 
```
```
>>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
tensor([[ 0],
        [ 1],
        [ 2],
        [ 4]])
>>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                [0.0, 0.4, 0.0, 0.0],
                                [0.0, 0.0, 1.2, 0.0],
                                [0.0, 0.0, 0.0,-0.4]]))
tensor([[ 0,  0],
        [ 1,  1],
        [ 2,  2],
        [ 3,  3]])
```
接下来介绍一个必定经常用到的函数，reshape。该函数用于改变tensor的形状，和其他的reshape一样，参数设为-1时，该维度的元素个数由其他维度计算得到。直接上示例：
```
>>> a = torch.arange(4)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])
>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])
```
有些时候不需要分割tensor，但是需要压缩tensor，pytorch提供了一个自动去掉通道是1的维度的函数：
```
torch.squeeze(input, dim=None, out=None)
```
直接上例子吧，比如我们生成一个维度为3，1，2的tensor
```
>>> x = torch.randn(3, 1, 2)
>>> x
tensor([[[-0.2863,  0.8594]],

        [[-0.4789,  0.9160]],

        [[ 1.0955, -1.2205]]])
```
可以发现实际上他的第二个维度没有什么意义。调用squeeze函数：
```
>>> torch.squeeze(x)
tensor([[-0.2863,  0.8594],
        [-0.4789,  0.9160],
        [ 1.0955, -1.2205]])
```
如果一个tensor中通道数为1的维度有很多，但是又不想全部去掉，那么可以在函数中通过设定dim参数，选择去掉某一个维度。
有的时候又需要增加维度，这时可以使用unsqueeze函数：
```
torch.unsqueeze(input, dim, out=None) 
```
该函数将在制定的维度增加一维：
```
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```
squeeze和unsqueeze函数产生的输出和输入都是共享存储空间的，改变其中一个另外一个也会改变。

既然tensor是矩阵操作，那么肯定少不了矩阵的转置：
```
torch.t(input, out=None)
```
例子很简单，函数也只有一个参数，那就是输入tensor：
```
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.4875,  0.9158, -0.5872],
        [ 0.3938, -0.6929,  0.6932]])
>>> torch.t(x)
tensor([[ 0.4875,  0.3938],
        [ 0.9158, -0.6929],
        [-0.5872,  0.6932]])
```
前面介绍过取部分元素的函数，这里再增加一个，用起来比较简单，但是在使用时可能人为需要计算的就多一点了：
```
torch.take(input, indices)
```
该函数将输入tensor转换为一个一维的向量，然后在该向量上依据给出的坐标，返回元素的tensor：
```
>>> src = torch.tensor([[4, 3, 5],
                        [6, 7, 8]])
>>> torch.take(src, torch.tensor([0, 2, 5]))
tensor([ 4,  5,  8])
```
pytorch除了提供了.t()这种简单的矩阵转置方式，还提供了另外一个函数：
```
torch.transpose(input, dim0, dim1, out=None)
```
这里通过指定两个维度dim0和dim1，来将其做转换，如果设为0和1则和t的效果等价。不过需要注意的是transpose函数转置前后的到的tensor是共享底层存储空间的，如果对其中一个的元素做更改，另外一个也会发生变化：
```
>>> x = torch.randn(3, 1, 2)
>>> x
tensor([[[-0.2863,  0.8594]],

        [[-0.4789,  0.9160]],

        [[ 1.0955, -1.2205]]])
>>> torch.squeeze(x)
tensor([[-0.2863,  0.8594],
        [-0.4789,  0.9160],
        [ 1.0955, -1.2205]])
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6525,  0.1151, -0.0437],
        [ 0.2640, -1.2813,  1.3332]])
>>> y = torch.transpose(x, 0, 1)
>>> y
tensor([[ 0.6525,  0.2640],
        [ 0.1151, -1.2813],
        [-0.0437,  1.3332]])
>>> y[0][0] = 1
>>> y
tensor([[ 1.0000,  0.2640],
        [ 0.1151, -1.2813],
        [-0.0437,  1.3332]])
>>> x
tensor([[ 1.0000,  0.1151, -0.0437],
        [ 0.2640, -1.2813,  1.3332]])
```
如果需要删除一个维度，可以用下面的操作：
```
torch.unbind(tensor, dim=0)
```
直接上示例吧：
```
>>> x
tensor([[ 1.0000,  0.1151, -0.0437],
        [ 0.2640, -1.2813,  1.3332]])
>>> torch.unbind(x, 1)
(tensor([ 1.0000,  0.2640]), tensor([ 0.1151, -1.2813]), tensor([-0.0437,  1.3332]))
```
可以看出，当我们删除了维度1之后，原tensor被分为了几个小的只有维度0的子tensor，所以这个函数可以简答的理解为按给定的维度将原tensor展开，上面是按列展开。再比如我们按行展开：
```
>>> torch.unbind(x, 0)
(tensor([ 1.0000,  0.1151, -0.0437]), tensor([ 0.2640, -1.2813,  1.3332]))
```
终于要把这一部分写完了，还剩最后一个函数：
```
torch.where(condition, x, y)
```
这个函数的功能是依据一个判断语句，来从tensor中挑选语句，官方文档给的例子是：
```
>>> x = torch.randn(3, 2)
>>> y = torch.ones(3, 2)
>>> x
tensor([[-0.4620,  0.3139],
        [ 0.3898, -0.7197],
        [ 0.0478, -0.1657]])
>>> torch.where(x > 0, x, y)
tensor([[ 1.0000,  0.3139],
        [ 0.3898,  1.0000],
        [ 0.0478,  1.0000]])
```
那么能不能依据x的值在y和z之中挑选呢？是可以的，甚至其他的条件都可以，但是有一个条件是你所使用的条件和变量是广播的。
```
>>> x = torch.randn(3, 2)
>>> y = torch.ones(3, 2)
>>> z = torch.zeros(3, 2)
>>> x
tensor([[ 0.9749,  0.2215],
        [-0.3449,  2.2324],
        [ 0.8020,  0.7957]])
>>> y
tensor([[ 1.,  1.],
        [ 1.,  1.],
        [ 1.,  1.]])
>>> z
tensor([[ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.]])
>>> torch.where(x>0,x,y)
tensor([[ 0.9749,  0.2215],
        [ 1.0000,  2.2324],
        [ 0.8020,  0.7957]])
```
#Random sampling
写到这里，我这一篇博客才写了四分之一(T_T)
随机数是深度学习中很关键的基础，为什么？自行百度吧。pytorch中设置随机数生成的种子点，查看种子点的函数分别为：
```
torch.manual_seed(seed) #设置
torch.initial_seed() #查看
```
示例如下：
```
>>> torch.initial_seed()
16846182053669541300
>>> torch.manual_seed(1)
<torch._C.Generator object at 0x7fadc8d9afb0>
>>> torch.initial_seed()
1
```
因为这篇博客里不打算探究pytorch随机数的生成原理，所以以下几个函数就不详细介绍了：get_rng_state()；set_rng_state(new_state)；torch.default_generator = <torch._C.Generator object>。
下面来介绍一系列随机tensor的生成方法
```
torch.bernoulli(input, out=None)
```
该函数按照伯努利分布依据输入的tensor来随机的生成0，1二值的tensor。要求输入tensor的元素值在0和1之间。
```
>>> a = torch.empty(3, 3).uniform_(0, 1) # generate a uniform random matrix with range [0, 1]
>>> a
tensor([[ 0.1737,  0.0950,  0.3609],
        [ 0.7148,  0.0289,  0.2676],
        [ 0.9456,  0.8937,  0.7202]])
>>> torch.bernoulli(a)
tensor([[ 1.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]])

>>> a = torch.ones(3, 3) # probability of drawing "1" is 1
>>> torch.bernoulli(a)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
>>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
>>> torch.bernoulli(a)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
```
```
torch.multinomial(input, num_samples, replacement=False, out=None) 
```
这个函数比较有意思，他的输入tensor表示了每一个位置的概率，这里并不要求他是大于1的，但是所有元素的和不能是0。num_samples为挑选出的位置的编号，实际上就是一个随机数，第三个参数指的是是否运行重复挑选。官方文档给的例子似乎不太好体现，给一个二值的输入tensor演示一下吧：
```
>>> weights = torch.tensor([0, 10], dtype=torch.float)
>>> torch.multinomial(weights, 4, replacement=True)
tensor([ 1,  1,  1,  1])
```
当讲两个位置的概率设为相同时：
```
>>> weights = torch.tensor([10, 10], dtype=torch.float)
>>> torch.multinomial(weights, 4, replacement=True)
tensor([ 1,  1,  0,  0])
```
注意如果后面的replacement设为false，那么输出的元素个数必须是不大于输入的size的。

下面一个方法生成的是服从正太分布的随机数：
有三种种方式：
```
torch.normal(mean, std, out=None) 
torch.normal(mean=0.0, std, out=None) 
torch.normal(mean, std=1.0, out=None)
```
如果输入的mean和std是tensor，那么生成的tensor将按照mean的形状，元素的个数和输入相同，也就是说mean和std可以有不同的形状，但是必须有相同的元素个数：
```
>>> torch.normal(mean=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))
tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
          8.0505,   8.1408,   9.0563,  10.0566])
```
另外也可以固定其中一个，也就是第二种和第三种方式：
```
>>> torch.normal(mean=0.5, std=torch.arange(1, 6))
tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])
>>> torch.normal(mean=torch.arange(1, 6))
tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])
```
下面几个功能类似，都是按照一定得规则生成随机tensor的，这里统一整理一下：
```python
#生成[0，1）的随机数
torch.rand(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
#按照输入的tensor的尺寸生成
torch.rand_like(input, dtype=None, layout=None, device=None, requires_grad=False) 
#在一个范围内生成整型的随机
torch.randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) 
#不解释
torch.randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False)
#返回01正太分布
torch.randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
#不解释
torch.randn_like(input, dtype=None, layout=None, device=None, requires_grad=False)
#返回0到输入n的之间整数随机排列不含n
torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
```
另外在torch.Tensor下还定义了一些in-place的函数：

*   [`torch.Tensor.bernoulli_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.bernoulli_ "torch.Tensor.bernoulli_") - in-place version of [`torch.bernoulli()`](https://pytorch.org/docs/stable/torch.html#torch.bernoulli "torch.bernoulli")，伯努利分布
*   [`torch.Tensor.cauchy_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.cauchy_ "torch.Tensor.cauchy_") - numbers drawn from the Cauchy distribution，柯西分布
*   [`torch.Tensor.exponential_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.exponential_ "torch.Tensor.exponential_") - numbers drawn from the exponential distribution，指数分布
*   [`torch.Tensor.geometric_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.geometric_ "torch.Tensor.geometric_") - elements drawn from the geometric distribution，几何分布
*   [`torch.Tensor.log_normal_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.log_normal_ "torch.Tensor.log_normal_") - samples from the log-normal distribution，对数正太分布
*   [`torch.Tensor.normal_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.normal_ "torch.Tensor.normal_") - in-place version of [`torch.normal()`](https://pytorch.org/docs/stable/torch.html#torch.normal "torch.normal")，正太分布
*   [`torch.Tensor.random_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.random_ "torch.Tensor.random_") - numbers sampled from the discrete uniform distribution，均匀分布
*   [`torch.Tensor.uniform_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.uniform_ "torch.Tensor.uniform_") - numbers sampled from the continuous uniform distribution，连续均匀分布
#Serialization
这里主要介绍如何将得到的tensor保存到本地，或者从本地读取tensor，这也是很关键的步骤。
```
torch.save(obj, f, pickle_module=<module 'pickle' from '/private/home/soumith/anaconda3/lib/python3.6/pickle.py'>, pickle_protocol=2)
```
这一部分在[Recommended approach for saving a model](https://pytorch.org/docs/stable/notes/serialization.html#recommend-saving-models)
有更详细的介绍。这里就放几个简单的例子吧：
```
>>> # Save to file
>>> x = torch.tensor([0, 1, 2, 3, 4])
>>> torch.save(x, 'tensor.pt')
>>> # Save to io.BytesIO buffer
>>> buffer = io.BytesIO()
>>> torch.save(x, buffer)
```
同样load函数：
```
>>> torch.load('tensors.pt')
# Load all tensors onto the CPU
>>> torch.load('tensors.pt', map_location='cpu')
# Load all tensors onto the CPU, using a function
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Load all tensors onto GPU 1
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
# Map tensors from GPU 1 to GPU 0
>>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
# Load tensor from io.BytesIO object
>>> with open('tensor.pt') as f:
        buffer = io.BytesIO(f.read())
>>> torch.load(buffer)
```
这一部分可能后续会专门写一篇博客介绍。这篇博客主要介绍tensor的相关操作函数吧。
#Parallelism
这里主要是和CPU线程相关的两个函数：
```
torch.get_num_threads()
torch.set_num_threads()
```
第一个用于的到目前的CPU线程，第二个用于设定使用的线程，但是在GPU训练和测试时基本没有什么用了。
#Locally disabling gradient computation
前面有提到过，pytorch会默认的给tensor计算梯度，这样在实际使用时，会增加一些不必要的资源开销，可以通过设置torch.no_grad(), torch.enable_grad()和torch.set_grad_enabled()三个值来设定tensor的梯度计算：
直接上例子吧：
```
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False

>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True

>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False
```
以上基本介绍完了tensor的各种基本的函数。下面步入这篇博客的正题。。。。写了快八千字了才步入正题。。。。
#Math operations
##Pointwise Ops
这一块将介绍在tensor上可以使用的各种数学计算。首先是最简单的各种运算，输入都是一个tensor，所以就不详细写了，还是直接整理一下：

求绝对值
```
torch.abs(input, out=None)
>>> torch.abs(torch.tensor([-1, -2, 3]))
tensor([ 1,  2,  3])
```
求反三角函数
```
torch.acos(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
>>> torch.acos(a)
tensor([ 1.2294,  2.2004,  1.3690,  1.7298])


torch.asin(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([-0.5962,  1.4985, -0.4396,  1.4525])
>>> torch.asin(a)
tensor([-0.6387,     nan, -0.4552,     nan])


torch.atan(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
>>> torch.atan(a)
tensor([ 0.2299,  0.2487, -0.5591, -0.5727])


torch.atan2(input1, input2, out=None) #输入为两个tensor，求他们对应的反正切
>>> a = torch.randn(4)
>>> a
tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
>>> torch.atan2(a, torch.randn(4))
tensor([ 0.9833,  0.0811, -1.9743, -1.4151])
```
所有元素加上一个定值
```
torch.add()
>>> a = torch.randn(4)
>>> a
tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
>>> torch.add(a, 20)
tensor([ 20.0202,  21.0985,  21.3506,  19.3944])
```
输入tensor加上一个常数乘以另一个tensor，如果两个维度不同，会扩展：
```
>>> a = torch.randn(4)
>>> a
tensor([-0.9732, -0.3497,  0.6245,  0.4022])
>>> b = torch.randn(4, 1)
>>> b
tensor([[ 0.3743],
        [-1.7724],
        [-0.5811],
        [-0.8017]])
>>> torch.add(a, 10, b)
tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
        [-18.6971, -18.0736, -17.0994, -17.3216],
        [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
        [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
```
对tensor1和tensor2做数除，然后乘以一个变量之后加到输入tensor上。
```
>>> t = torch.randn(1, 3)
>>> t1 = torch.randn(3, 1)
>>> t2 = torch.randn(1, 3)
>>> torch.addcdiv(t, 0.1, t1, t2)
tensor([[-0.2312, -3.6496,  0.1312],
        [-1.0428,  3.4292, -0.1030],
        [-0.5369, -0.9829,  0.0430]])
```
类似的也有乘法：
```
torch.addcmul(tensor, value=1, tensor1, tensor2, out=None) 
```
还有一个减法的，但是好像物理意义不太一样，公式如下：
![lerp](https://upload-images.jianshu.io/upload_images/11609151-9ae35a9829c0c164.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
torch.lerp(start, end, weight, out=None)
>>> start = torch.arange(1, 5)
>>> end = torch.empty(4).fill_(10)
>>> start
tensor([ 1.,  2.,  3.,  4.])
>>> end
tensor([ 10.,  10.,  10.,  10.])
>>> torch.lerp(start, end, 0.5)
tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
```
对tensor中的元素做向上取整？
```
torch.ceil(input, out=None) 

>>> a = torch.randn(4)
>>> a
tensor([-0.6341, -1.4208, -1.0900,  0.5826])
>>> torch.ceil(a)
tensor([-0., -1., -1.,  1.])
```
向下取整：
```
torch.floor(input, out=None) 

>>> a = torch.randn(4)
>>> a
tensor([-0.8166,  1.5308, -0.2530, -0.2091])
>>> torch.floor(a)
tensor([-1.,  1., -1., -1.])
```
x大于或小于阈值时将其截断：
```
torch.clamp(input, min, max, out=None) 

>>> a = torch.randn(4)
>>> a
tensor([-1.7120,  0.1734, -0.0478, -0.0922])
>>> torch.clamp(a, min=-0.5, max=0.5)
tensor([-0.5000,  0.1734, -0.0478, -0.0922])
#当然也可以只设置一边
>>> a = torch.randn(4)
>>> a
tensor([-0.0299, -2.3184,  2.1593, -0.8883])
>>> torch.clamp(a, min=0.5)
tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

>>> a = torch.randn(4)
>>> a
tensor([ 0.0753, -0.4702, -0.4599,  0.1899])
>>> torch.clamp(a, max=0.5)
tensor([ 0.0753, -0.4702, -0.4599,  0.1899])
```
三角函数
```
#余弦
torch.cos(input, out=None)
>>> a = torch.randn(4)
>>> a
tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
>>> torch.cos(a)
tensor([ 0.1395,  0.2957,  0.6553,  0.5574])

#双曲余弦
torch.cosh(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
>>> torch.cosh(a)
tensor([ 1.0133,  1.7860,  1.2536,  1.2805])
```
元素数除法，可以除数也可以除tensor
```
torch.div()
>>> a = torch.randn(5)
>>> a
tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
>>> torch.div(a, 0.5)
tensor([ 0.7620,  2.5548, -0.5944, -0.7439,  0.9275])

>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
        [ 0.1815, -1.0111,  0.9805, -1.5923],
        [ 0.1062,  1.4581,  0.7759, -1.2344],
        [-0.1830, -0.0313,  1.1908, -1.4757]])
>>> b = torch.randn(4)
>>> b
tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
>>> torch.div(a, b)
tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
        [ 0.2260, -3.4507, -1.2086,  6.8988],
        [ 0.1322,  4.9764, -0.9564,  5.3480],
        [-0.2278, -0.1068, -1.4678,  6.3936]])
```
计算元素除法的各项余数：
```
torch.fmod(input, divisor, out=None) 
>>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
tensor([-1., -0., -1.,  1.,  0.,  1.])
>>> torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
```
元素乘法：
```
torch.mul()

>>> a = torch.randn(3)
>>> a
tensor([ 0.2015, -0.4255,  2.6087])
>>> torch.mul(a, 100)
tensor([  20.1494,  -42.5491,  260.8663])

>>> a = torch.randn(4, 1)
>>> a
tensor([[ 1.1207],
        [-0.3137],
        [ 0.0700],
        [ 0.8378]])
>>> b = torch.randn(1, 4)
>>> b
tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
>>> torch.mul(a, b)
tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
        [-0.1614, -0.0382,  0.1645, -0.7021],
        [ 0.0360,  0.0085, -0.0367,  0.1567],
        [ 0.4312,  0.1019, -0.4394,  1.8753]])
```

计算误差函数：
![误差函数](https://upload-images.jianshu.io/upload_images/11609151-a1e6fae5fed974bf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
torch.erf(tensor, out=None)
>>> torch.erf(torch.tensor([0, -1., 10.]))
tensor([ 0.0000, -0.8427,  1.0000])
```
计算反误差函数：
![反误差函数](https://upload-images.jianshu.io/upload_images/11609151-0bba080bbe7cd219.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
torch.erfinv(tensor, out=None)
>>> torch.erfinv(torch.tensor([0, 0.5, -1.]))
tensor([ 0.0000,  0.4769,    -inf])
```
指数函数：
```
torch.exp(tensor, out=None)
>>> torch.exp(torch.tensor([0, math.log(2)]))
tensor([ 1.,  2.])
```
归零化的指数函数（求指数后减一）
```
torch.expm1(tensor, out=None)
>>> torch.expm1(torch.tensor([0, math.log(2)]))
tensor([ 0.,  1.])
```
计算元素的小数部分：
```
torch.frac(tensor, out=None) 
>>> torch.frac(torch.tensor([1, 2.5, -3.2]))
tensor([ 0.0000,  0.5000, -0.2000])
```
下面是一堆求log的函数：
```
# 自然对数
torch.log(input, out=None)
#10为底
torch.log10(input, out=None) 
#自然对数，输入为1+input
torch.log1p
#以2为底
torch.log2(input, out=None)
```
取反：
```
torch.neg(input, out=None)
>>> a = torch.randn(5)
>>> a
tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
>>> torch.neg(a)
tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
```
求指数：
```
torch.pow()

>>> a = torch.randn(4)
>>> a
tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
>>> torch.pow(a, 2)
tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
>>> exp = torch.arange(1, 5)

>>> a = torch.arange(1, 5)
>>> a
tensor([ 1.,  2.,  3.,  4.])
>>> exp
tensor([ 1.,  2.,  3.,  4.])
>>> torch.pow(a, exp)
tensor([   1.,    4.,   27.,  256.])

>>> exp = torch.arange(1, 5)
>>> base = 2
>>> torch.pow(base, exp)
tensor([  2.,   4.,   8.,  16.])
```
求倒数：
```
torch.reciprocal(input, out=None)
>>> a = torch.randn(4)
>>> a
tensor([-0.4595, -2.1219, -1.4314,  0.7298])
>>> torch.reciprocal(a)
tensor([-2.1763, -0.4713, -0.6986,  1.3702])
```
求余数：
```
torch.remainder(input, divisor, out=None)
>>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
>>> torch.remainder(torch.tensor([1., 2, 3, 4, 5]), 1.5)
tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
```
四舍五入：
```
torch.round(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
>>> torch.round(a)
tensor([ 1.,  1.,  1., -1.])
```
去掉小数部分：
```
torch.trunc(input, out=None)
>>> a = torch.randn(4)
>>> a
tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
>>> torch.trunc(a)
tensor([ 3.,  0., -0., -0.])
```
平方根倒数：
```
torch.rsqrt(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([-0.0370,  0.2970,  1.5420, -0.9105])
>>> torch.rsqrt(a)
tensor([    nan,  1.8351,  0.8053,     nan])
```
sigmoid函数，这个用的就相当多了：
![sigmoid](https://upload-images.jianshu.io/upload_images/11609151-3dd850d91470b44f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
torch.sigmoid(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
>>> torch.sigmoid(a)
tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
```
符号函数，正为1负为-1：
```
torch.sign(input, out=None) 
>>> a = torch.randn(4)
>>> a
tensor([ 1.0382, -1.4526, -0.9709,  0.4542])
>>> torch.sign(a)
tensor([ 1., -1., -1.,  1.])
```
另外还有正弦函数sin（），反三角函数sinh（），平方根函数sqrt（），正切函数tan（），反正切函数tanh（）等就不再一一给出例子了。

##Reduction Ops
按指定的维度返回最大元素的坐标。
```
torch.argmax(input, dim=None, keepdim=False)
>>> a = torch.randn(4, 4)
>>> a
tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])


>>> torch.argmax(a, dim=1)
tensor([ 0,  2,  0,  1])
```
同样，返回最小的：
```
torch.argmin(input, dim=None, keepdim=False)
```
累乘：
```
torch.cumprod(input, dim, out=None) 
>>> a = torch.randn(10)
>>> a
tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
        -0.2129, -0.4206,  0.1968])
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
         0.0014, -0.0006, -0.0001])

>>> a[5] = 0.0
>>> torch.cumprod(a, dim=0)
tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
         0.0000, -0.0000, -0.0000])
```
累加：
```
torch.cumsum(input, dim, out=None) 
>>> a = torch.randn(10)
>>> a
tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
         0.1850, -1.1571, -0.4243])
>>> torch.cumsum(a, dim=0)
tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
        -1.8209, -2.9780, -3.4022])
```
返回两个tensor之间差的（input - other）的p范数。这个在距离度量里面应该也是用的很普遍的。
```
torch.dist(input, other, p=2)
>>> x = torch.randn(4)
>>> x
tensor([-1.5393, -0.8675,  0.5916,  1.6321])
>>> y = torch.randn(4)
>>> y
tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
>>> torch.dist(x, y, 3.5)
tensor(1.6727)
>>> torch.dist(x, y, 3)
tensor(1.6973)
>>> torch.dist(x, y, 0)
tensor(inf)
>>> torch.dist(x, y, 1)
tensor(2.6537)
```
返回一个tensor向量的p范数：
```
torch.norm(input, p=2)
>>> a = torch.randn(1, 3)
>>> a
tensor([[-0.5192, -1.0782, -1.0448]])
>>> torch.norm(a, 3)
tensor(1.3633)
torch.norm(input, p, dim, keepdim=False, out=None)
>>> a = torch.randn(4, 2)
>>> a
tensor([[ 2.1983,  0.4141],
        [ 0.8734,  1.9710],
        [-0.7778,  0.7938],
        [-0.1342,  0.7347]])
>>> torch.norm(a, 2, 1)
tensor([ 2.2369,  2.1558,  1.1113,  0.7469])
>>> torch.norm(a, 0, 1, True)
tensor([[ 2.],
        [ 2.],
        [ 2.],
        [ 2.]])
```
求均值：
```
torch.mean(input)
>>> a = torch.randn(1, 3)
>>> a
tensor([[ 0.2294, -0.5481,  1.3288]])
>>> torch.mean(a)
tensor(0.3367)
torch.mean(input, dim, keepdim=False, out=None)
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
        [-0.9644,  1.0131, -0.6549, -1.4279],
        [-0.2951, -1.3350, -0.7694,  0.5600],
        [ 1.0842, -0.9580,  0.3623,  0.2343]])
>>> torch.mean(a, 1)
tensor([-0.0163, -0.5085, -0.4599,  0.1807])
>>> torch.mean(a, 1, True)
tensor([[-0.0163],
        [-0.5085],
        [-0.4599],
        [ 0.1807]])
```
还有求中位数median（）用法和mean是一样的。还有求众数mode（）。
不过这两个的返回值不仅包含了要求的值，还有原tensor中的坐标：
```
>>> a = torch.randn(4, 5)
>>> a
tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
        [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
        [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
        [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
>>> torch.median(a, 1)
(tensor([-0.3982,  0.2270,  0.2488,  0.4742]), tensor([ 1,  4,  4,  3]))
```
返回一个tensor中所有元素的乘积：
```
torch.prod(input) 
>>> a = torch.randn(1, 3)
>>> a
tensor([[-0.8020,  0.5428, -1.5854]])
>>> torch.prod(a)
tensor(0.6902)
torch.prod(input, dim, keepdim=False, out=None) 
>>> a = torch.randn(4, 2)
>>> a
tensor([[ 0.5261, -0.3837],
        [ 1.1857, -0.2498],
        [-1.1646,  0.0705],
        [ 1.1131, -1.0629]])
>>> torch.prod(a, 1)
tensor([-0.2018, -0.2962, -0.0821, -1.1831])
```
和这种形式类似的还有标准差std（），求和sum（），方差var（）。

剔除tensor中的重复元素，如果设置return_inverse=True，会得到一个元素在在原tensor中的映射表：
```
torch.unique(input, sorted=False, return_inverse=False)
>>> output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
>>> output
tensor([ 2,  3,  1])

>>> output, inverse_indices = torch.unique(
        torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
>>> output
tensor([ 1,  2,  3])
>>> inverse_indices
tensor([ 0,  2,  1,  2])

>>> output, inverse_indices = torch.unique(
        torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
>>> output
tensor([ 1,  2,  3])
>>> inverse_indices
tensor([[ 0,  2],
        [ 1,  2]])
```


未完待续。。。



























