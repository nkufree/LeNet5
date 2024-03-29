import numpy as np
from math import exp

class Conv:
    def __init__(self, input_shape, kernel_size, kernel_num, stride=1) -> None:
        self.input_shape = input_shape
        C, H, W = input_shape # 个数、通道数、高度、宽度
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.stride = stride
        self.filters = np.random.uniform(-1,1,(kernel_num, C, kernel_size, kernel_size))
        self.bias = np.random.uniform(-1, 1, kernel_num)
        row = (H - self.kernel_size) // stride + 1
        col = (W - self.kernel_size) // stride + 1
        self.output_shape = (self.kernel_num, row, col)
        # self.shape1 = (N, C, row, col, kernel_size, kernel_size)
    
    def forward(self, input):
        # self.strides1 = (*input.strides, *input.strides[-2:])
        # 对矩阵进行分片
        self.input:np.ndarray = input
        self.input_shape = input.shape
        C, H, W = self.input_shape
        _, row, col = self.output_shape
        split_input = self.split_by_stride(input,(C, row, col, self.kernel_size, self.kernel_size))
        '''
        使用爱因斯坦求和约定计算卷积
        i -> 通道数
        j -> 输出的行数
        k -> 输出的列数
        p -> 卷积核个数（输出通道数）
        l -> 卷积核的行
        m -> 卷积核的列
        '''
        self.output = np.einsum('ijklm,pilm->pjk', split_input, self.filters)\
            + self.bias.repeat(row * col).reshape(self.output_shape)
        self.output = 1 / (1 + np.exp(-self.output))
        self.output_shape = self.output.shape
        return self.output
    
    def backprop(self, input: np.ndarray, alpha):
        input = input.reshape(self.output.shape)
        # 乘以sigmoid激活函数的偏导
        input = self.output * (1 - self.output) * input
        # 对输入求梯度
        C, H, W = self.output_shape
        in_c, in_h, in_w = self.input_shape
        remain_h = (H - self.kernel_size) % self.stride
        remain_w = (W - self.kernel_size) % self.stride
        pad_top = self.kernel_size - 1
        pad_bottom = self.kernel_size - 1 + remain_h
        pad_left = self.kernel_size - 1
        pad_right = self.kernel_size - 1 + remain_w
        padded_input = np.pad(input, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right)), 'constant', constant_values=0)
        row = (H + pad_top + pad_bottom - self.kernel_size) // self.stride + 1
        col = (W + pad_left + pad_right - self.kernel_size) // self.stride + 1
        split_padded_input = self.split_by_stride(padded_input, (C, row, col, self.kernel_size, self.kernel_size))
        input_grade = np.einsum('ijklm,iplm->pjk', split_padded_input, np.rot90(self.filters, 2, (2,3)))
        # 对权重求梯度，更新权重
        split_net_input = self.split_by_stride(self.input, (in_c, self.kernel_size, self.kernel_size, H, W))
        weight_grade: np.ndarray = np.einsum('ijklm,plm->pijk', split_net_input, input)
        self.filters -= weight_grade * alpha
        # 对偏置求梯度，更新偏置项
        bias_grade = np.einsum('cij->c', input)
        self.bias -= bias_grade * alpha
        return input_grade
    
    def split_by_stride(self, input, shape):
        strides = (*input.strides[:-2], input.strides[-2]*self.stride, input.strides[-1]*self.stride, *input.strides[-2:])
        return np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
