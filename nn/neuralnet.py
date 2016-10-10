import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers

    def add(self, layers):
        if type(layers) == list:
            self.layers.extend(layers)
        else: 
            self.layers.append(layers)
        
    def predict(self, x):
        pass

    def loss(self, x, y_true):
        input = x
        for layers in self.layers:
            input = layers.fwd_prop(input)
        y_pred = input
        return self.layers[-1].loss(y_true, y_pred)

    def save(self, file_path):
        layers = [{"W": self.layers.W, "b": self.layers.b} for layer in self.layers]
        np.save(file_path, layers)

    def restore(self, file_path):
        layers = np.load(file_path)
        for i in range(self.layers):
            self.layers[i].W = layers[i]["W"]
            self.layers[i].b = layers[i]["b"]

    def input_error(self, y):
        return y - self.layers[-1].output