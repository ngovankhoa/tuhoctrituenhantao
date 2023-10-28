""" 
Copyright (C) 2023 Ngo Van Khoa.
"""
import mnist
from neural_net2 import NeuralNet2

# nạp dữ liệu
data_train, data_test = mnist.load()

# tạo mạng neuron
net = NeuralNet2((784, 16, 10))

# huấn luyện
learning_rate = 0.9
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