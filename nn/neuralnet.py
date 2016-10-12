import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers

    def add(self, layers):
        if type(layers) == list:
            self.layers.extend(layers)
        else: 
            self.layers.append(layers)
    
    def fit(x, y_true, epochs=3, learning_rate=0.1, batch_size=128):
        batches = len(x)/batch_size

        for epoch in range(epochs):
            for batch in range(batches):
                start_batch_idx = batch * batch_size
                end_batch_idx = start_batch_idx + batch_size

                x_batch = x[start_batch_idx:end_batch_idx]
                y_batch = y[start_batch_idx:end_batch_idx]

                #forward pass
                y_pred = self.predict(x_batch)

                
    def predict(self, x):
        input = x
        for layer in self.layers:
            input = layer.fwd_prop(input)
        y_pred = input
        return y_pred

    def loss(self, x, y_true, loss_function):
        y_pred = self.predict(x)
        if type(loss_function) == str:
            loss_function = getattr(self, loss_function)
        return loss_function(y_true, y_pred)

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

    def cross_entropy(self, y_true, y_pred):
        return np.mean(-np.sum(y_true * np.log(y_pred)))

    def MSE(self, target):
        return np.mean((self.layers[-1].output - target)**2)

    def MAE(self, target):
        return np.mean(np.abs(self.layers[-1].output - target))