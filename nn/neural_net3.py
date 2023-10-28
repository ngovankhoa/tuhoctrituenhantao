""" 
NeuralNet3 -- Dùng Cross Entropy 
Copyright (C) 2023 Ngo Van Khoa.
"""
from neural_net2 import NeuralNet2
import random
import mnist
import numpy as np
class NeuralNet3(NeuralNet2):

    def delta_last_layer(self, a, y, z):
        """
        Tính delta ở lớp cuối cùng theo (3.5)
        """
        return (a - y)