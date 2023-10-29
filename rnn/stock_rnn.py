"""
NeuralNet -- Dự đoán giá chứng khoán với RNN
Copyright (C) 2023 Ngo Van Khoa.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#nạp dữ liệu
data = np.loadtxt('../data/apple.txt')

#chuẩn hóa dữ liệu với MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

#tổ chức dữ liệu:
#chúng ta sẽ chia dữ liệu huấn luyện là các mảng 60 
#phần tử là giá đóng cửa của 60 ngày liên tiếp
#và nhãn là giá đóng cửa của ngày thứ 61
time_len = 60
X = []
y = []
for i in range(len(data) - time_len):
    X.append(data[i:i + time_len])
    y.append(data[i + time_len])
print("X len:", len(X), "y len:", len(y))

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

#tạo mô hình
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
model = Sequential([
    Input(shape=(time_len, 1)),
    SimpleRNN(128),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

#huấn luyện và dự đoán giá dựa vào dữ liệu test
model.fit(X_train, y_train, epochs = 50, batch_size =32)
y_pred = model.predict(X_test)

#vẽ đồ thị so sánh
import matplotlib.pyplot as plt
y_pred = scaler.inverse_transform(y_pred)
y_test = y_test.reshape((-1, 1))
y_test = scaler.inverse_transform(y_test)
y_test = np.reshape(y_test, (-1, 1))
y_pred = np.reshape(y_pred,(-1, 1))
plt.figure(figsize = (30,10))
plt.plot(y_pred, ls='--', label='Giá dự đoán', lw = 2)
plt.plot(y_test, label='Giá thực tế')
plt.legend()
plt.show()