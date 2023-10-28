""" 
NeuralNet2 -- Mạng neuron thần kinh nhân tạo có thể học
Copyright (C) 2023 Ngo Van Khoa.
"""
from neural_net import NeuralNet
import random
import mnist
import numpy as np
class NeuralNet2(NeuralNet):

    def __init__(self, layers):
        super().__init__(layers) 
    
    def sigmoid_derivative(self, z):
        """
        Tính đạo hàm của hàm sigmoid
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def delta_last_layer(self, a, y, z):
        """
        Tính delta ở lớp cuối cùng theo (2.17)
        """
        return (a - y) * self.sigmoid_derivative(z)
    
    def backpropagation(self, x, y):
        """
        Thuật toán lan truyền ngược dùng để tính đạo hàm của hàm
        mất mát theo w và b
        """
        # a là danh sách chứa các vectơ đầu ra ở mỗi lớp
        # với lớp đầu vào, a1 chính là x
        a = [x]
        z = []

        # feedforward, giữ lại z và a để tính đạo hàm
        a_i = x
        for i in range(self.L - 1):            
            z_i = np.dot(self.w[i], a_i) + self.b[i]            
            a_i = self.sigmoid(z_i)
            z.append(z_i)            
            a.append(a_i)

        # Tính đạo hàm riêng của hàm mất mát cho từng w và b        
        derivative_w = []
        derivative_b = []

        # tính đạo hàm ở lớp cuối cùng
        d_L =  self.delta_last_layer(a[-1], y, self.sigmoid_derivative(z[-1]))
        derivative_b_L = d_L # theo (2.22)
        derivative_w_L =  np.dot(d_L, a[-2].transpose()) # theo (2.23)
        derivative_b.append(derivative_b_L)
        derivative_w.append(derivative_w_L)

        # tính đạo hàm ở các lớp ẩn
        d_l_1 = d_L 
        for l in reversed(range(1, self.L - 1)):
            # Chú ý cách đánh dấu chỉ mục trong Python và trong mô hình mạng neuron
            # Chúng ta không lưu trữ dư thừa, vì vậy, mô hình có L lớp, thì các mảng
            # w, b và z sẽ sẽ chỉ có chiều dài L - 1. Vì vậy, khi truy cập l trong w, b
            # và z từ Python chính là lớp thứ l + 1 trong mô hình 

            # tính delta l
            z_derivative = self.sigmoid_derivative(z[l - 1]) # z[l - 1] là z[l] trong mô hình
            
            # w[l] chính là w[l + 1] trong mô hình            
            d_l =  np.dot(self.w[l].transpose(), d_l_1) * z_derivative # (2.24)

            derivative_b_l = d_l # theo (2.25)
            derivative_w_l = np.dot(d_l, a[l-1].transpose()) # theo (2.26)
            derivative_b.insert(0, derivative_b_l)
            derivative_w.insert(0, derivative_w_l)

            # d_l sẽ trở thành d_l_1 cho các lớp trước
            d_l_1 = d_l

        return (derivative_w, derivative_b)

    def train(self, X, Y, learning_rate):
        """
        Huấn luyện mạng theo dữ liệu X và nhãn Y
        """                        
        derivative_w_batch = [np.zeros(w_i.shape) for w_i in self.w]             
        derivative_b_batch = [np.zeros(b_i.shape) for b_i in self.b]

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            # với mỗi x, tính đạo hàm
            derivative_w, derivative_b = self.backpropagation(x, y)

            # cộng vào đạo hàm tổng            
            for i in range(len(derivative_w_batch)):
                derivative_w_batch[i] = derivative_w_batch[i] + derivative_w[i]
            for i in range(len(derivative_b_batch)):
                derivative_b_batch[i] = derivative_b_batch[i] + derivative_b[i]

        # tính trung bình đạo hàm (2.21)
        m = len(X)
        for i in range(len(derivative_w_batch)):
            derivative_w_batch[i] = derivative_w_batch[i] / m
        for i in range(len(derivative_b_batch)):
            derivative_b_batch[i] = derivative_b_batch[i] / m

        # cập nhật lại w và b theo (2.4) và (2.5)
        for i in range(len(derivative_w_batch)):
            self.w[i] = self.w[i] - learning_rate * derivative_w_batch[i]
        for i in range(len(derivative_b_batch)):
            self.b[i] = self.b[i] - learning_rate * derivative_b_batch[i]


    def SGD(self, data_train, learning_rate, batch_size, num_epoch, data_test=None):
        """
        Huấn luyện mạng theo thuật toán Stochastic Gradient Descent (SGD).
        *data_train* là dữ liệu huấn luyện        
        *learning_rate* là tốc độ học
        *batch_size* là kích thước của tập con
        *num_epoch* là số kỷ nguyên cần huấn luyện
        *data_test* là dữ liệu nằm ngoài *x_train*, được xem như là dữ liệu mới,
            nhằm kiểm tra kết quả của việc huấn luyện        
        """        
        n = len(data_train[0])

        # huấn luyện lặp lại với nhiều kỷ nguyên
        for epoch in range(num_epoch):
            print("*** Đang huấn luyện kỷ nguyên #{}...".format(epoch))
            # trộn đều danh sách vì vậy chúng ta có thể chọn tập con với các phần tử ngẫu nhiên
            X_train, Y_train = mnist.suffle(data_train)

            # huấn luyện trên từng tập con
            for i in range(0, n, batch_size):
                X_batch = X_train[i: i + batch_size]
                Y_batch = Y_train[i: i + batch_size]
                self.train(X_batch, Y_batch, learning_rate)

            # đánh giá kết quả huấn luyện nếu có dữ liệu *data_test*
            if data_test is not None:
                self.evaluate(data_test)
    
    def evaluate(self, data_test):
        X = data_test[0]
        Y = data_test[1]        
        correct = 0
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            a = self.inference(x)            
            output = mnist.output_to_digit(a)
            label = mnist.output_to_digit(y)
            if output == label:
                correct = correct + 1
        print("Độ chính xác: {:.2f}%".format(100 * correct / len(X)))