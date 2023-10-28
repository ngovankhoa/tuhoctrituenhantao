from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

# nạp dữ liệu để huấn luyện
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# chuyển ma trận thành vectơ
x_train = x_train.reshape(-1, 784).astype('float32') #(60000, 784)
x_test = x_test.reshape(-1, 784).astype('float32')

# chuẩn hóa dữ liệu
x_train = x_train/255
x_test = x_test/255

# chuyển nhãn thành vectơ
y_train = utils.to_categorical(y_train, num_classes=10) #(60000, 10)
y_test = utils.to_categorical(y_test, num_classes=10)

import tensorflow as tf

# tạo một mạng neuron nhân tạo
model = tf.keras.models.Sequential()

# thêm lớp đầu vào
layer = tf.keras.layers.Input(shape=(784))
model.add(layer)

# thêm vào lớp ẩn 64 neuron
layer = tf.keras.layers.Dense(64, activation='sigmoid')
model.add(layer)

# thêm lớp đầu ra 10 neuron
layer = tf.keras.layers.Dense(10, activation='sigmoid')
model.add(layer)

# hàm mất mát
loss_fn = tf.keras.losses.BinaryCrossentropy()

# thuật toán huấn luyện
sgd = tf.keras.optimizers.SGD(learning_rate=0.2)

# cấu hình mô hình
model.compile(optimizer=sgd, loss=loss_fn, metrics=['accuracy'])

# huấn luyện mô hình
model.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))