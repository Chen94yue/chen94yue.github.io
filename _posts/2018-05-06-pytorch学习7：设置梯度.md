深度学习很重要的一点是反向传播，所以梯度的计算尤为重要，pytorch中对于tensor可以自动的计算梯度，对于其中已经包含的层，梯度的计算也是自动的。但是当你使用一个fine-tunning的操作，或者自己定义层时，如何定义处理梯度问题，或者设置梯度就尤为关键了。这里将简单介绍一下pytorch中如何开启或关闭tensor的梯度计算和层的梯度计算。
直接上代码：
```
>>> x = torch.randn(5, 5)  # requires_grad=False by default
>>> y = torch.randn(5, 5)  # requires_grad=False by default
>>> z = torch.randn((5, 5), requires_grad=True)
>>> a = x + y
>>> a.requires_grad
False
>>> b = a + z
>>> b.requires_grad
True
```
再pytorch的tensor生成函数randn中设置参数requires_grad能够开启或者关闭该tensor的梯度计算。从上面的结果可以看出梯度的计算是传递的，虽然a没有梯度，但是z有梯度，所以b也有了。

另外对于设置层的梯度计算，有下面的例子：
```
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
```
这里主要是在fine-tunning的场合用的很多，首先使用torchvision加载一个预训练模型，resnet18，并且加载他的预训练参数。在fine-tunning的时候，我们不需要重新训练其上面的层的参数，因此我们设置其上的每一层的requires_grad为false。之后在其上增加一个fc层，会默认设置该层的requires_grad的为true。下面设置学习方法，SGD，并且只需要设置新增加的fc层的学习参数即可。
