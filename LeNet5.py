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
        self.add(Conv((1,32,32),5,6))\
            .add(Subsampling((6,28,28),2))\
            .add(Conv((6,14,14),5,16))\
            .add(Subsampling((16,10,10),2))\
            .add(Conv((16,5,5),5,120))\
            .add(Connect(120,84))\
            .add(Output(84,10))
        # self.add(Connect(1024,84))\
        #     .add(Output(84,10))
    
    def add(self, net):
        self.networks.append(net)
        return self
    
    def forward(self, input):
        output = input
        for net in self.networks:
            output = net.forward(output)
        return output
    
    def backprop(self, input, alpha=0.1):
        output = input
        for net in reversed(self.networks):
            output = net.backprop(output, alpha)
    
    def predict(self, input):
        ret = []
        for entry in input:
            ret.append(self.forward(entry))
        preds = np.argmax(ret, axis=1)
        return preds
    
    def train(self, train_data: np.ndarray, y, iter=100, alpha=0.1):
        
        loss = []
        total_num = train_data.shape[0]
        for i in range(iter):
            for j in range(total_num // 10):
                score = self.forward(train_data[j]).T
                # print(score)
                score -= np.max(score)
                exp_score = np.exp(score)
                p = exp_score / np.sum(exp_score)
                func = np.frompyfunc(lambda x:x if x != 0 else 1e-20,1,1)
                p = func(p).astype(np.float32).T
                f = -np.sum(y[j]*np.log(p))
                loss.append(f)
                print(f)
                self.backprop(p-y[j] , alpha)