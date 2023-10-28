""" 
Copyright (C) 2023 Ngo Van Khoa.
"""
import mnist
from neural_net4 import NeuralNet4

# nạp dữ liệu
data_train, data_test = mnist.load()
# lấy 10000 phần tử đầu tiên
train_x = data_train[0][:10000]
train_y = data_train[1][:10000]
data_train = (train_x, train_y)

# tạo mạng neuron
#net = NeuralNet2((784, 16, 10))
#net = NeuralNet3((784, 64, 10))
net = NeuralNet4((784, 64, 10))

# huấn luyện
learning_rate = 0.5
batch_size = 10
num_epoch = 40

net.SGD(data_train, learning_rate, batch_size, num_epoch, data_test)

# cho mạng nhận diện
for i in range(10):
    x = data_test[0][i]
    a = net.inference(x)
    y = data_test[1][i]    
    a_out = mnist.output_to_digit(a)
    y_out = mnist.output_to_digit(y)
    print('Mạng nhận diện {}, thực tế là {}'.format(a_out, y_out))