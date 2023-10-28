import mnist
import neural_net
import numpy as np

# tạo mạng neuron
net = neural_net.NeuralNet((784, 16, 10))

# load dữ liệu
data_train, data_test = mnist.load()

# cho mạng thử suy luận
for i in range(5):
    x = data_train[0][i]
    a = net.inference(x)
    y = data_train[1][i]    
    a_out = mnist.output_to_digit(a)
    y_out = mnist.output_to_digit(y)
    print('Mạng nhận diện {}, thực tế là {}'.format(a_out, y_out))