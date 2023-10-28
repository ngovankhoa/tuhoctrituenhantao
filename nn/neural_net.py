"""
NeuralNet -- Mạng thần kinh
Copyright (C) 2023 Ngo Van Khoa.
"""
import numpy as np
class NeuralNet:
    def __init__(self, layers):
        """Khởi tạo một mạng thần kinh với số lớp và kích thước mỗi lớp được
        chỉ định trong tuple layers. Ví dụ, để khởi tạo một mạng neuron với
        3 lớp, lớp thứ nhất có 3 neuron, lớp thứ 2 có 4 neuron, lớp thứ 3
        có 2 neuron, ta truyền vào layers = (3,4,2).
        """
        self.layers = layers
        self.L = len(layers)
        # Khởi tạo các ma trận w
        self.w = list()
        for i in range(1, self.L):
            n = layers[i]
            m = layers[i - 1]
            # Tạo một ma trận n hàng m cột với các giá trị ngẫu nhiên
            w_i = np.random.randn(n, m)
            #print("{}x{}".format(n, m))
            #print(w_i)
            self.w.append(w_i)

        # Khởi tạo các vecto cột (ma trận nhiều hàng một cột) b
        self.b = list()
        for i in range(1, self.L):
            n = layers[i]
            b_i = np.random.randn(n, 1)
            self.b.append(b_i)

    def sigmoid(self, z):
        """
        Hàm kích hoạt sigmoid. Bạn có thể truyền vào một số,
        một vecto hoặc một ma trận. Kết quả trả về sẽ là sigmoid
        của số hoặc vector hay ma trận các sigmoid cho từng phần tử
        của vector hay ma trận được truyền vào.
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def inference(self, x):
        """
        Với đầu vào x là vecto cột self.layers[0] phần tử,
        hàm inference trả về kết quả suy luận từ mạng thần kinh
        là một vecto cột self.layers[self.L - 1] phần tử
        """
        a = x
        for i in range(self.L - 1):
            #print('==========================================')
            #print('a = ')
            #print(a)
            z = np.dot(self.w[i], a) + self.b[i]
            #print('z = ')
            #print(z)
            a = self.sigmoid(z)
            #print('a = ')
            #print(a)
        return a
    
'''
net = NeuralNet((3,4,2))
x = np.matrix([[6],[8],[2]]) #vectơ cột có 3 phần tử
a = net.inference(x)
print(a)
'''