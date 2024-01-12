import numpy as np
from math import exp

class Subsampling:
    def __init__(self, input_shape, kernel_size, stride=2) -> None:
        C, H, W = input_shape # 个数、通道数、高度、宽度
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.kernel_num = C
        self.stride = stride
        self.filters = np.random.uniform(-1, 1, C)
        self.bias = np.random.uniform(-1, 1, C)
        self.row = (H - self.kernel_size) // stride + 1
        self.col = (W - self.kernel_size) // stride + 1
        self.output_shape = (C, self.row, self.col)
    
    def forward(self, input):
        # 对矩阵进行分片
        self.input = input
        self.input_shape = input.shape
        self.split_input = self.split_by_stride(input, (input.shape[0], self.row, self.col, self.kernel_size, self.kernel_size))
        # 使用爱因斯坦求和约定计算下采样
        C, row, col = self.output_shape
        self.output = np.einsum('ijklm,i->ijk', self.split_input, self.filters)\
            + self.bias.repeat(row * col).reshape(self.output_shape)
        # sigmoid = np.frompyfunc(lambda x:0 if x < -100 else 1/(1+exp(-x)),1,1)
        # self.output = sigmoid(self.output).astype(np.float32)
        self.output = 1 / (1 + np.exp(-self.output))
        self.output_shape = self.output.shape
        return self.output
        # output = np.zeros(self.output_shape, dtype=np.float32)
        # for i in range(self.output_shape[0]):
        #     for j in range(self.output_shape[1]):
        #         for k in range(self.output_shape[2]):
        #             output[i][j][k] = np.sum(input[i][j:j+self.kernel_size,k:k+self.kernel_size]) * self.weight[i] + self.bias[i]
        # return output

    def backprop(self, input: np.ndarray, alpha):
        input = input.reshape(self.output_shape)
        # 乘以sigmoid激活函数的偏导
        input = self.output * (1 - self.output) * input
        # 对输入的导数
        C, H, W = self.output_shape
        in_c, in_h, in_w = self.input_shape
        input_grade:np.ndarray = np.einsum('cij,c->cij', input, self.filters)
        input_grade = input_grade.repeat(self.kernel_size, axis=1).repeat(self.kernel_size, axis=2)
        # 对权重的导数
        # split_net_input = self.split_by_stride(self.input, (in_c, self.kernel_size, self.kernel_size, H, W))
        # weight_grade = np.einsum('ijklm,clm->c', split_net_input, input)
        weight_grade = np.einsum('crlij,crl->c', self.split_input, input)
        self.filters -= weight_grade * alpha
        # 对偏置项的导数
        bias_grade = np.einsum('cij->c', input)
        self.bias -= bias_grade * alpha
        return input_grade
    
    def split_by_stride(self, input, shape):
        strides = (*input.strides[:-2], input.strides[-2]*self.stride, input.strides[-1]*self.stride, *input.strides[-2:])
        return np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)