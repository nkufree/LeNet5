import numpy as np
from conv import Conv, Conv3
from subsampling import Subsampling
from connect import Connect
from output import Output
from math import log
import random

class LeNet5:
    networks = []
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        # self.add(Conv((batch_size, 1,32,32),5,6))\
        #     .add(Subsampling((batch_size, 6,28,28),2))\
        #     .add(Conv((batch_size, 6,14,14),5,16))\
        #     .add(Subsampling((batch_size, 16,10,10),2))\
        #     .add(Conv((batch_size, 16,5,5),5,120))\
        #     .add(Connect(batch_size, 120,84))\
        #     .add(Output(batch_size, 84,10))
        self.add(Connect(batch_size, 1024,84))\
            .add(Output(batch_size, 84,10))
    
    def add(self, net):
        self.networks.append(net)
        return self
    
    def calc(self, input):
        output = input
        for net in self.networks:
            output = net.calc(output)
        return output
    
    def update(self, input, alpha=0.1):
        output = input
        for net in reversed(self.networks):
            output = net.update(output, alpha)
    
    def predict(self, input):
        ret = self.calc(input)
        preds = np.argmax(ret, axis=1)
        return preds
    
    def train(self, train_data: np.ndarray, y, iter=100, alpha=0.1):
        
        loss = []
        total_num = train_data.shape[0]
        for i in range(iter):
            randList = random.sample(range(0,total_num), self.batch_size)
            score = self.calc(train_data[randList]).T
            print(score)
            score -= np.max(score, axis=0)
            exp_score = np.exp(score)
            p = exp_score / np.sum(exp_score, axis=0)
            func = np.frompyfunc(lambda x:x if x != 0 else 1e-20,1,1)
            p = func(p).astype(np.float32).T
            f = -np.sum(y[randList]*np.log(p)) / self.batch_size
            loss.append(f)
            print(f)
            self.update(np.sum(p-y[randList], axis=0) / self.batch_size, alpha)