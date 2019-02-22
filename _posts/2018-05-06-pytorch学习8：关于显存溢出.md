```
cuda runtime error(2): out of memory
```
上面这一行应该是最常见的bug之一了。解决这个问题，可以按以下步骤：
第一，检查一下你的显卡的存储是否满足实验要求。
——这里的实验要求不太好界定，基本跑的主干网络多了，应该能有一个大致的估计，vgg16输入224*224的图像batchsize设为多少能占多少显存，resnet50又能占多少。然后看看自己的网络结构，大概需要多少显存，如果大于了显卡的显存容量，一般可以选择的操作就是减小batchsize了。如果batchsize设为1还是不能满足基本需求，那基本可以选择放弃了。
第二，检查一下自己的网络结构中是否有没有必要的缓存。这里pytorch的官方文档举了一个很好的例子。如下：
```
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss
```
这里在计算total_loss时将每次迭代产生的loss累加。pytorch有一个默认的设置，就是如果没有设置的话，每一个从网络得到的tensor是含有一个自动计算的梯度量的。多以total_loss不仅仅累加了loss的值，还累积了loss的梯度。为了避免这一部分的显存开销，只需要在累加之前做一步类型变换：
```
total_loss += float(loss)
```
另一种情况是中间变量太多，官网指出中间变量在计算过程中不会自动释放，除非del掉，因此减少不必要的中间变量也是很关键的一步，如下面的例子：
```
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
output = h(result)
return output
```
显存中会同时保存着intermediate，result和output。其实上面面的语句用一行就可以解决，或者当你用完之后del掉不需要的变量。
另外不要使用太长的RNN和太大的FC也是很关键的一点。如果FC太大，可以考虑写成卷积的形式，卷积核的大小设为feature map的大小，卷积核的个数设为需要的FC输出的向量长度。
第三，检查是否后台有没有关闭的其他占用显存的程序正在运行。有时候服务器上同时运行着多个实验，有时候上一个实验跑完或者bug了，但是显存并没有释放，进行新的实验时很容易出现这个情况。最简单的方式是使用nvidia-smi看一下现在显存占用的情况，和使用显存的程序号，或者使用ps -elf | grep python查询（grep过滤了不是python的程序，如果有运行基于c语言或者matlab的深度学习代码，该指令将不会显示），然后kill掉它（kill -9 [pid]）

