import numpy as np
from math import exp

class Connect:
    def __init__(self, bantch_size, input_size, output_size) -> None:
        self.bantch_size = bantch_size
        self.input_size = input_size
        self.input_shape = (bantch_size, input_size)
        self.output_size = output_size
        self.output_shape = (bantch_size, output_size)
        self.weight = np.random.uniform(-1, 1, (input_size, output_size))
        self.bias = np.random.uniform(-1, 1, output_size)
    
    def calc(self, input: np.ndarray):
        self.input = input.reshape((input.shape[0], -1,))
        self.output = np.einsum('ni,ij->nj',self.input,self.weight)\
            + self.bias.reshape((1, self.output_size))
        # output = np.zeros(self.output_size, dtype=np.float32)
        # for i in range(self.output_size):
        #     for j in range(self.input_size):
        #         output[i] += self.weight[i][j] * self.input[j]
        #     output[i] += self.bias[i]
        #     output[i] = 1/(1+exp(-output[i]))
        sigmoid = np.frompyfunc(lambda x:0 if x < -100 else 1/(1+exp(-x)),1,1)
        self.output = sigmoid(self.output).astype(np.float32)
        return self.output
    
    def update(self, input: np.ndarray, alpha):
        input = input.reshape(*self.output.shape[1:])
        # 乘以sigmoid激活函数的偏导
        input = self.output * (1 - self.output) * input
        # 计算三者梯度
        input_grade = np.einsum('nj,ij->ni', input, self.weight)
        weight_grade = np.einsum('ni,nj->ij', self.input, input)
        bias_grade = np.sum(input, axis=0)
        self.weight -= weight_grade * alpha / self.bantch_size
        self.bias -= bias_grade * alpha / self.bantch_size
        return input_grade.reshape((self.output.shape[0], self.input_size,1))


