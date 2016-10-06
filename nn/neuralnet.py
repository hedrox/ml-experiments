import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers


    def input_error(self, y):
        return y - self.layers[-1].output