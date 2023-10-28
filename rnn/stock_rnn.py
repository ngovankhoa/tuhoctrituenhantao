import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import Input

#nạp dữ liệu
data = np.loadtxt('../data/apple.txt')

#chuẩn hóa dữ liệu với MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

#tổ chức dữ liệu:
#chúng ta sẽ chia dữ liệu huấn luyện là các mảng 60 
#phần tử là giá đóng cửa của 60 ngày liên tiếp
#và nhãn là giá đóng cửa của ngày thứ 61
time_len = 60
X = []
y = []
for i in range(len(scaled_data) - time_len):
    X.append(scaled_data[i:i + time_len])
    y.append(scaled_data[i + time_len])

print("X:", len(X), "y:", len(y))
   
#tách tập dữ liệu test ra riêng tỉ lệ 80:20
train_len = round(len(X) * 0.8)
X_train = X[:train_len] 
X_test = X[train_len:]
y_train = y[:train_len]
y_test = y[train_len:]

#định dạng ma trận để làm việc với mô hình
X_train = np.asarray(X_train).reshape((-1, time_len, 1))
y_train = np.asarray(y_train)
X_test = np.asarray(X_test).reshape((-1, time_len, 1))
y_test = np.asarray(y_test)

print("X shape:", X_train.shape, "y shape:", y_train.shape)

model = Sequential()

model.add(Input(shape=(time_len, 1)))
model.add(SimpleRNN(128))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


history = model.fit(X_train, y_train, epochs = 50, batch_size =32)

y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

y_test = y_test.reshape((-1, 1))
y_test = scaler.inverse_transform(y_test)

y_test = np.reshape(y_test, (-1, 1))
y_pred = np.reshape(y_pred,(-1, 1))
plt.figure(figsize = (30,10))
plt.plot(y_pred, ls = '--', label = 'Giá dự đoán', lw = 2)
plt.plot(y_test, label = 'Giá thực tế')
plt.legend()
plt.show()

