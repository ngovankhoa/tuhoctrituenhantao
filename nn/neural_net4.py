""" 
NeuralNet4 -- Đánh giá mô hình
Copyright (C) 2023 Ngo Van Khoa.
"""
from neural_net3 import NeuralNet3
import numpy as np
import mnist

class NeuralNet4(NeuralNet3):

    def loss(self, a, y):        
        """
        Trả về giá trị của hàm mất mát theo (3.2)        
        """
        # Nếu cả a và y có giá trị 1.0 tại một hàng nào đó
        # thì (1-y)*np.log(1-a) sẽ trả về NaN.
        # Chúng ta dùng np.nan_to_num để chuyển NaN thành 0
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def loss_total(self, data):
        """
        Tính loss trên toàn bộ tập dữ liệu, theo (3.1)
        """
        lost = 0
        n = len(data[0])
        for i in range(n):
            x = data[0][i]
            y = data[1][i]
            a = self.inference(x)
            lost = lost + self.loss(a, y)
        return lost/n
    
    def SGD(self, data_train, learning_rate, batch_size, num_epoch, data_test=None):
        """
        Viết lại hàm SGD để thêm các thông tin ước lượng mô hình
        """        
        n = len(data_train[0])

        # huấn luyện lặp lại với nhiều kỷ nguyên
        for epoch in range(num_epoch):
            print("*** Đang huấn luyện kỷ nguyên #{}...".format(epoch))            
            X_train, Y_train = mnist.suffle(data_train)

            # huấn luyện trên từng tập con
            for i in range(0, n, batch_size):
                X_batch = X_train[i: i + batch_size]
                Y_batch = Y_train[i: i + batch_size]
                self.train(X_batch, Y_batch, learning_rate)

            # đánh giá kết quả huấn luyện            
            self.evaluate(data_train, data_test)

    def accuracy(self, data):
        """
        Trả về độ chính xác trên tập dữ liệu
        """ 
        n = len(data[0])
        correct = 0.0
        for i in range(n):
            x = data[0][i]
            y = data[1][i]
            a = self.inference(x)            
            output = mnist.output_to_digit(a)
            label = mnist.output_to_digit(y)
            if output == label:
                correct = correct + 1
        return 100.0 * correct / n
    
    def evaluate(self, data_train, data_test):
        loss_train = self.loss_total(data_train)
        loss_test = self.loss_total(data_test)
        accuracy_train = self.accuracy(data_train)
        accuracy_test = self.accuracy(data_test)
        print("\tĐộ chính xác train: {:.2f}%\tĐộ mất mát train: {:.2f}"
              .format(accuracy_train, loss_train))
        print("\tĐộ chính xác  test: {:.2f}%\tĐộ mất mát  test: {:.2f}"
              .format(accuracy_test, loss_test))