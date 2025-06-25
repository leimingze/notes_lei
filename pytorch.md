# 张量的常见操作
## 张量的基本创建与属性
```py
import torch
a=torch.tensor([1,2,3,4,5])
print(f"一维张量{a}")
print(f"形状{a.shape}")

b=torch.tensor([[1,2,3],[4,5,6]])
print(f"二维张量{b}")
print(f"形状{b.shape}")

print(torch.ones(2,3))
print(torch.eye(3))
```
## 张量的常见运算
```py
import torch
a=torch.tensor([1,2,3])
b=torch.tensor([10,20,30])
print("加法",a+b)
print("乘法",a*b)
print("平方",a**2)
print("求和",torch.sum(a))
#每个元素+5
mat = torch.rand(5, 5)
mat = mat + 5   # 或 mat += 5
print(mat)

#矩阵乘法
mat1=torch.rand(2,3)
mat2=torch.rand(3,4)
res=torch.matmul(mat1,mat2)
print("矩阵乘法",res)
```
## 张量与numpy的转换
```py
import torch
import numpy as np
arr=np.array([1,2,3,4,5])
tensor_from_np=torch.from_numpy(arr)
print("tensor_from_np: ",tensor_from_np)

np_from_tensor=tensor_from_np.numpy()
print("np_from_tensor: ",np_from_tensor)
```
# 自动求导与梯度
## 什么是自动求导/梯度
自动求导，只要你用张量.requires_grad=True，pytorch会自动追踪每一步操作帮你算导数（梯度
梯度：指每个参数改多少才能让损失变少。训练神经网络时，就是靠梯度来更新参数的
requires_grad=True：表示让pytorch追踪这个张量的每一步操作，从而自动求导，计算梯度
```py
import torch

x=torch.tensor([2.0],requires_grad=True)#必须是浮点数
#因为只有浮点数才能标识连续的数轴，才能做无限小的变化。
y=x**2+3*x+1
y.backward()#自动求导

print("x的值",x.item())
print("y的值",y.item())
print("x的梯度",x.grad.item())
#x.item()：把只包含一个元素的张量转换成python普通数字
#x.grad.item()：把张量的梯度转换成python普通数字 x.grad是张量
# x的值 2.0
# y的值 11.0
# x的梯度 7.0
```
## 多步操作和梯度清零
```py
import torch
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])
y = w * x + b  # 线性函数

y.backward()   # 只对 w 和 b 求导

print("w的梯度:", w.grad)   # x 相当于y对w求偏导
print("b的梯度:", b.grad)   # 1 相当于y对b求偏导
```
```py
import torch
from torch import nn

# 1. 准备数据 (这里用100个点, 加点噪声)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # shape [100, 1]
y = 2 * x + 3 + 0.2 * torch.rand(x.size())              # 加噪声, 更真实

# 2. 定义模型 (1层线性网络)
model = nn.Linear(1, 1)  # 输入1维, 输出1维

# 3. 定义损失函数 (均方误差)
loss_fn = nn.MSELoss()

# 4. 定义优化器 (随机梯度下降)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 5. 训练循环
for epoch in range(100):
    pred = model(x)             # 前向传播
    loss = loss_fn(pred, y)     # 计算损失

    optimizer.zero_grad()       # 梯度清零
    loss.backward()             # 自动求导
    optimizer.step()            # 用梯度更新参数

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# 6. 查看模型学到的参数
w, b = model.weight.item(), model.bias.item()
print(f"学到的参数: w={w:.3f}, b={b:.3f}")

# 问题1 为什么要加一维?为什么是dim=1?
#答：因为有多少行表示有多少个样本，有多少列标识有多少个特征，所以这里加一维表示有100个样本，每个样本只有一个特征，所以是1维。
# 问题2 为什么是torch.rand(x.size())?
# 答：因为torch.rand(x.size())是随机生成一个与x相同维度的张量，然后与y相加，这样生成的y就不再是常量了，而是与x相关的随机数，这样训练出来的模型就更加真实了。
#如果rand(1)，那么y就是100个常量，那么训练出来的模型就是一条直线，与x无关，这样训练出来的模型就不够真实了。
#问题3 定义模型时输入1维, 输出1维, 是什么意思?
#答输入一维就是每个样本只有一个特征，输出一维标识每个样本只有一个结果，所以输入输出都是1维。

```