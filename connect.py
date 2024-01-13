import numpy as np
from math import exp

class Connect:
    def __init__(self, input_size, output_size) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.uniform(-1, 1, (input_size, output_size))
        self.bias = np.random.uniform(-1, 1, output_size)
    
    def forward(self, input: np.ndarray):
        self.input = input.reshape((-1,))
        self.output = np.einsum('i,ij->j',self.input,self.weight)\
            + self.bias
        self.output = 1 / (1 + np.exp(-self.output))
        return self.output
    
    def backprop(self, input: np.ndarray, alpha):
        input = input.reshape(self.output.shape)
        # 乘以sigmoid激活函数的偏导
        input = self.output * (1 - self.output) * input
        # 计算三者梯度
        input_grade = np.einsum('j,ij->i', input, self.weight)
        weight_grade = np.einsum('i,j->ij', self.input, input)
        bias_grade = np.sum(input, axis=0)
        self.weight -= weight_grade * alpha
        self.bias -= bias_grade * alpha
        return input_grade.reshape((self.input_size,1))


