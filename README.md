# LeNet5网络实现

南开大学 机器学习（谢晋老师） EX2（大作业）2023年秋

## 实验要求

在这个练习中，需要使用`Python`实现`LeNet5`来完成对MNIST数据集中0-9共10个手写数字的分类。代码只能使用Python实现，其中数据读取可使用PIL、opencv-python等库，矩阵运算可使用`numpy`等计算库，网络前向传播和梯度反向传播需要手动实现，不能使用PyTorch、TensorFlow、Jax或Caffe等自动微分框架。MNIST数据集可在 http://yann.lecun.com/exdb/mnist/下载。

## 网络结构

`LeNet5`网络共有七层：

1. 卷积层，使用6个大小为5×5的卷积核。
   + 输入：1×32×32（分别代表通道数、高度、宽度。下同）
   + 输出：6×28×28
2. 池化层，使用1个大小为2×2的卷积核，步长为2。
   + 输入：6×28×28
   + 输出：6×14×14
3. 卷积层，使用16个大小为5×5的卷积核。
   + 输入：6×14×14
   + 输出：16×10×10
4. 池化层，使用1个大小为2×2的卷积核， 步长为2。
   + 输入：16×10×10
   + 输出：16×5×5
5. 卷积层，使用120个大小为5×5的卷积核。
   + 输入：16×5×5
   + 输出：120×1×1
6. 全连接层。
   + 输入：120
   + 输出：84
7. 输出层，使用径向基函数。
   + 输入：84
   + 输出：10

除了输出层，每层之后都要加上`sigmoid`激活函数。

在此基础上，我们在最后一层之后加上了`softmax`和交叉熵损失函数，用于优化输出结果。

其实吧，我感觉输出层应该可以直接换成`softmax`，但是为了尽可能复现原版的网络结构，在此保留径基函数的实现。
