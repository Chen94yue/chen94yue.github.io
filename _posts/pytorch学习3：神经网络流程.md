现在试着用pytorch搭建一个手写字母识别的网络，这是一个很经典的demo，网络结构如下：
![网络结构](http://upload-images.jianshu.io/upload_images/11609151-d1a662f54db07a9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
流程包括以下几步
1.定义一个神经网络
2.迭代输入训练数据
3.前向传播
4.计算loss
5.反向传播
6.更新网络参数（weight = weight - learning_rate * gradient，weight往梯度下降的方向增加）

首先定义网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()
print(net)
```
输出为：
```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```
可以通过调用.parameters()来查看其中一层的参数：
```
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```
输出为
```
10
torch.Size([6, 1, 5, 5])
```

我们尝试给网络一个随机的输入：
```
input = torch.randn(1, 1, 32, 32)
```
这里四个参数定义了一个四维的矩阵，如果按灰度图像来看，第一个相当于图像的张数，第二个相当与图像的通道数，灰度为1，rgb为3，后面两个相当于图像的尺寸。
通过网络：
```
out = net(input)
print(out)
```
之后可以得到：
```
tensor([[-0.0089, -0.0514,  0.0059,  0.1412, -0.1543,  0.0494, -0.0966,
         -0.1150, -0.0986, -0.1103]])
```

下面计算loss，这里使用MSEloss：
```
output = net(input)
target = torch.arange(1, 11)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```
得到输出：
```
tensor(39.2273)
```
整个网络的前向过程如下所示：
```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```
通过使用loss.backward()函数实现梯度的反向传播：
```
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```
可以得到输出为：
```
conv1.bias.grad before backward
tensor([ 0.,  0.,  0.,  0.,  0.,  0.])
conv1.bias.grad after backward
tensor([ 0.0501,  0.1040, -0.1200,  0.0833,  0.0081,  0.0120])
```
关于pytorch中的各个层的详细介绍看[这里](https://pytorch.org/docs/stable/nn.html)

关于梯度下降算法，需要使用torch.optim包，如下所示：
```
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```
步骤为首先设置梯度下降的方式（支持SGD, Nesterov-SGD, Adam, RMSProp等），并设置学习率，在每一次迭代中首先清空梯度计算的缓存，然后输入计算数据，计算loss，反向传播，调用.step()完成参数的更新。



