from datasets import load_dataset
import numpy as np
import random

def transform_image(x):
    x = np.array(x)/255.0
    x = np.reshape(x, (784, 1))
    return x

def transform_label(y):
    y2 = np.zeros((10, 1))
    y2[y] = 1.0
    return y2

def load():
    mnist = load_dataset('mnist')

    X_train = [transform_image(x) for x in mnist['train']['image']]
    y_train = [transform_label(y) for y in mnist['train']['label']]
    data_train = (X_train, y_train)

    X_test = [transform_image(x) for x in mnist['test']['image']]
    y_test = [transform_label(y) for y in mnist['test']['label']]
    data_test = (X_test, y_test)

    return data_train, data_test

def output_to_digit(a):
    """
    Hàm này sẽ chuyển dữ liệu từ đầu ra thành số
    nhận diện được để chúng ta dễ đọc
    """
    return a.argmax(axis=0)[0]

def suffle(data):
    """
    Trộn một tuple *data* theo thứ tự ngẫu nhiên 
    nhưng vẫn giữ đúng nhãn của dữ liệu
    """
    temp = list(zip(data[0], data[1]))
    random.shuffle(temp)
    X, y = zip(*temp)    
    return (list(X), list(y))