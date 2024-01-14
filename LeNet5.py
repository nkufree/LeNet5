import numpy as np
from conv import Conv
from subsampling import Subsampling
from connect import Connect
from output import Output
from math import log
import random
import pickle

class LeNet5:
    networks = []
    def __init__(self, batch_size=6000) -> None:
        self.batch_size = batch_size
        self.add(Conv((1,32,32),5,6))\
            .add(Subsampling((6,28,28),2))\
            .add(Conv((6,14,14),5,16))\
            .add(Subsampling((16,10,10),2))\
            .add(Conv((16,5,5),5,120))\
            .add(Connect(120,84))\
            .add(Output(84,10))
    
    def dump(self, fp):
        # 将所有层训练得到的参数导出
        params = []
        # 1. 前五层导出filters和bias
        for i in range(5):
            params.append([self.networks[i].filters,self.networks[i].bias])
        # 2. 第六层导出weight和bias
        params.append([self.networks[5].weight, self.networks[5].bias])
        with open(fp, "wb+") as f:
            pickle.dump(params, f)
    
    def load(self, fp):
        with open(fp, "rb") as f:
            params = pickle.load(f)
        for i in range(5):
            self.networks[i].filters = params[i][0]
            self.networks[i].bias = params[i][1]
        
        self.networks[5].weight = params[5][0]
        self.networks[5].bias = params[5][1]
    
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
    
    def train(self, train_data: np.ndarray, y, iter=100, alpha=0.5):
        y_arg = np.argmax(y, axis=1)
        loss = []
        acc = []
        total_num = train_data.shape[0]
        for i in range(iter):
            curr_loss = 0
            num = 0
            for data, label in zip(train_data, y):
                if random.random() > self.batch_size / total_num:
                    continue
                num += 1
                # 前向传播
                score = self.forward(data).T
                # print(score)
                # 进行softmax处理
                score -= np.max(score)
                exp_score = np.exp(score)
                p = exp_score / np.sum(exp_score)
                func = np.frompyfunc(lambda x:x if x != 0 else 1e-20,1,1)
                # 计算交叉熵损失
                p = func(p).astype(np.float32).T
                f = -np.sum(label*np.log(p))
                curr_loss += f
                # print(f)
                # 反向传播
                self.backprop(p-label , alpha)
            # 保存平均损失率
            loss.append(curr_loss / num)
            # 计算在训练集上的准确率
            preds = self.predict(train_data)
            curr_acc = 0
            for a, b in zip(preds, y_arg):
                if a == b:
                    curr_acc += 1
            acc.append(curr_acc / total_num)
            print(f"epoch: {i+1}\t\tloss: {format(loss[-1], '.5f')}\tacc: {format(acc[-1], '.5f')}")
            # 学习率衰减
            alpha *= 0.95
        return loss, acc