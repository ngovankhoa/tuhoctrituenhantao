""" 
NeuralNet5 -- Chính quy hóa L2
Copyright (C) 2023 Ngo Van Khoa.
"""
from neural_net4 import NeuralNet4
import numpy as np
import mnist

class NeuralNet5(NeuralNet4):
    
    def train(self, X, Y, learning_rate, n, regularization_param = 0):
        """
        Viết làm hàm train với chính quy hóa L2
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

        # cập nhật lại w với chính quy hóa
        for i in range(len(derivative_w_batch)):
            self.w[i] = self.w[i] - learning_rate * \
                (derivative_w_batch[i] + (regularization_param/n) * self.w[i])
        for i in range(len(derivative_b_batch)):
            self.b[i] = self.b[i] - learning_rate * derivative_b_batch[i]

    def SGD(self, data_train, learning_rate, batch_size, num_epoch,\
            data_test=None, regularization_param=0):
        """
        Viết lại hàm SGD với chính quy hóa
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
                self.train(X_batch, Y_batch, learning_rate, n, regularization_param)

            # đánh giá kết quả huấn luyện            
            self.evaluate(data_train, data_test, regularization_param)

    def evaluate(self, data_train, data_test, regularization_param = 0):
        loss_train = self.loss_total(data_train)
        loss_test = self.loss_total(data_test)

        # Tính L2
        l2 = sum(np.linalg.norm(w)**2 for w in self.w)        
        loss_train += 0.5*(regularization_param/len(data_train[0]))*l2                
        loss_test += 0.5*(regularization_param/len(data_test[0]))*l2

        accuracy_train = self.accuracy(data_train)
        accuracy_test = self.accuracy(data_test)
        print("\tĐộ chính xác train: {:.2f}%\tĐộ mất mát train: {:.2f}"
              .format(accuracy_train, loss_train))
        print("\tĐộ chính xác  test: {:.2f}%\tĐộ mất mát  test: {:.2f}"
              .format(accuracy_test, loss_test))