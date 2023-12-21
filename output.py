import numpy as np
from PIL import Image

class Output:
    def __init__(self, bantch_size, input_size, output_size) -> None:
        self.batch_size = bantch_size
        self.input_size = input_size
        self.input_shape = (bantch_size, input_size)
        self.output_size = output_size
        self.output_shape = (bantch_size, output_size)
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
    
    def calc(self, input):
        self.input = input
        self.batch_size = input.shape[0]
        self.output = np.zeros(self.batch_size*self.output_size, dtype=np.float32).reshape((self.batch_size, self.output_size))
        # 由于按照RBF计算，越小越好，所以这里取相反数，使得结果越大越好
        for n in range(self.batch_size):
            for i in range(self.output_size):
                for j in range(self.input_size):
                    self.output[n][i] -= (input[n][j] - self.weight[i][j]) ** 2
        return self.output

    def update(self, input: np.ndarray, alpha):
        input = input.reshape(self.output_shape)
        # 对输入求梯度
        input_grade = np.zeros(self.input_shape, dtype=np.float32)
        for i in range(self.input_size):
            for j in range(self.output_size):
                    for n in range(self.batch_size):
                        input_grade[n][i] -= input[n][j] * 2 * (self.input[n][i] - self.weight[j][i])
        return input_grade