import numpy as np
import struct
import os
from LeNet5 import LeNet5
import json
import warnings

warnings.filterwarnings("ignore")

def load_mnist(file_dir, is_images='True'):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
    return mat_data

# call the load_mnist function to get the images and labels of training set and testing set
def load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir):
    print('Loading MNIST data from files...')
    train_images = load_mnist(os.path.join(mnist_dir, train_data_dir), True)
    train_labels = load_mnist(os.path.join(mnist_dir, train_label_dir), False)
    test_images = load_mnist(os.path.join(mnist_dir, test_data_dir), True)
    test_labels = load_mnist(os.path.join(mnist_dir, test_label_dir), False)
    return train_images, train_labels, test_images, test_labels


def train(net:LeNet5, train_images, y, iter=20, alpha=0.5):
    loss, acc = myNet.train(train_images, y, iter, alpha)
    with open("loss.json", "w+", encoding="utf-8") as f:
        json.dump(loss, f)
    with open("acc.json", "w+", encoding="utf-8") as f:
        json.dump(acc, f)

def test(net:LeNet5, test_images, test_labels):
    # 在测试集上测试
    answer = myNet.predict(test_images)
    # print(answer)
    # print(train_labels[0:5])
    # print(res)
    success_num = 0
    for a, b in zip(answer, test_labels):
        if a == b:
            success_num += 1
    print("在测试集上的准确率为：", success_num / test_images.shape[0])

mnist_dir = "mnist_data/"
train_data_dir = "train-images-idx3-ubyte"
train_label_dir = "train-labels-idx1-ubyte"
test_data_dir = "t10k-images-idx3-ubyte"
test_label_dir = "t10k-labels-idx1-ubyte"

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    train_images[train_images < 40] = 0
    train_images[train_images >= 40] = 1
    train_images = np.array([[np.pad(i.reshape(28,28), ((2,2),(2,2)), 'constant', constant_values=(0,0))] for i in train_images])
    test_images = np.array([[np.pad(i.reshape(28,28), ((2,2),(2,2)), 'constant', constant_values=(0,0))] for i in test_images])
    y = np.zeros((train_images.shape[0],10))
    for i in range(train_images.shape[0]):
        y[i,train_labels[i]] = 1
    batch_size = 6000
    myNet = LeNet5(batch_size)
    # 训练
    train(myNet, train_images, y)
    myNet.dump('model.pkl')
    # 测试
    myNet.load('model.pkl')
    test(myNet, test_images, test_labels)