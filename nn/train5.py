""" 
Copyright (C) 2023 Ngo Van Khoa.
"""
import mnist
from neural_net5 import NeuralNet5

# nạp dữ liệu
data_train, data_test = mnist.load()

# tạo mạng neuron
#net = NeuralNet2((784, 16, 10))
#net = NeuralNet3((784, 64, 10))
#net = NeuralNet4((784, 64, 10))
net = NeuralNet5((784, 100, 10))

# huấn luyện
learning_rate = 0.5
batch_size = 10
num_epoch = 100

net.SGD(data_train, learning_rate, batch_size, num_epoch, data_test, 
    regularization_param=5.0)

# cho mạng nhận diện
for i in range(10):
    x = data_test[0][i]
    a = net.inference(x)
    y = data_test[1][i]    
    a_out = mnist.output_to_digit(a)
    y_out = mnist.output_to_digit(y)
    print('Mạng nhận diện {}, thực tế là {}'.format(a_out, y_out))