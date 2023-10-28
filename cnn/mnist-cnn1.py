"""
Mô hình mnist với CNN dùng TensorFlow
Copyright (C) 2023 Ngo Van Khoa.
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf

# nạp dữ liệu để huấn luyện
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# chuẩn hóa dữ liệu
x_train = x_train/255
x_test = x_test/255

# chuyển nhãn thành vectơ
y_train = utils.to_categorical(y_train, num_classes=10) #(60000, 10)
y_test = utils.to_categorical(y_test, num_classes=10)

# xây dựng mô hình
input_shape = (28, 28, 1)
model = Sequential([
    # lớp đầu vào
    layers.Input(shape=input_shape),
    # 3 lớp tích chập
    layers.Conv2D(3, kernel_size=(5, 5), activation='sigmoid'),
    # lớp tổng hợp
    layers.MaxPooling2D(pool_size=(2, 2)), 
    # chuyển thành cột neuron để thực hiện lớp kết nối đầy đủ
    layers.Flatten(), 
    # lớp kết nối đầy đủ
    layers.Dense(10, activation='sigmoid')
])

# thuật toán huấn luyện
sgd = tf.keras.optimizers.SGD(learning_rate=0.2)
# cấu hình mô hình
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# huấn luyện mô hình
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))