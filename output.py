import numpy as np
from PIL import Image

class Output:
    def __init__(self, input_size, output_size) -> None:
        self.input_size = input_size
        self.output_size = output_size
        # self.weight = np.random.rand(output_size, input_size)
        self.load_weight('number.png')
    
    def load_weight(self, fp):
        im = Image.open(fp)
        arr = np.asarray(im)
        # 由于上一步输出的范围为0-1，所以这里将白色设置为0，黑色设置为1
        f = np.frompyfunc(lambda x:0 if x == 1 else 1, 1, 1)
        numbers = f(arr).astype(np.int32)
        split_number = np.split(numbers, 10, axis=1)
        for i in range(len(split_number)):
            split_number[i] = split_number[i].reshape((-1,))
        self.weight = np.array(split_number)
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_size, dtype=np.float32)
        # 由于按照RBF计算，越小越好，所以这里取相反数，使得结果越大越好
        for i in range(self.output_size):
            for j in range(self.input_size):
                self.output[i] -= (input[j] - self.weight[i][j]) ** 2
        return self.output

    def backprop(self, input: np.ndarray, alpha):
        input = input.reshape(self.output_size)
        # 对输入求梯度
        input_grade = np.zeros(self.input_size, dtype=np.float32)
        for i in range(self.input_size):
            for j in range(self.output_size):
                        input_grade[i] -= input[j] * 2 * (self.input[i] - self.weight[j][i])
        return input_grade