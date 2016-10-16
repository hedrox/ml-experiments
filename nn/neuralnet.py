import numpy as np
import h5py
import objectives

class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers

    def add(self, layers):
        if type(layers) == list:
            self.layers.extend(layers)
        else: 
            self.layers.append(layers)
    
    def fit(self, x, y_true, epochs=10000, learning_rate=0.1):
        #batches = len(x)/batch_size
        x_batch = x
        y_batch = y_true
        
        for epoch in range(epochs):
            for x_input in x_batch:
                #forward pass
                y_pred = self.predict(x_input)
                #backward pass
                out_error = self.output_error(y_batch, y_pred)
                delta = out_error * self.layers[-1].d_output(y_pred)
                self.layers[-1].delta = delta
                for layer in range(2,len(self.layers)):
                    delta = self.layers[-layer].bwd_prop(delta)

                #SGD
                for layer in self.layers[1:]:
                    print layer
                    layer.W += learning_rate * layer.a.dot(layer.delta)
            # for batch in range(batches):
            #     start_batch_idx = batch * batch_size
            #     end_batch_idx = start_batch_idx + batch_size

            #     x_batch = x[start_batch_idx:end_batch_idx]
            #     y_batch = y[start_batch_idx:end_batch_idx]

                
    def predict(self, x):
        input = x
        for layer in self.layers:
            input = layer.fwd_prop(input)
        y_pred = input
        return y_pred

    def loss(self, x, y_true, loss_function):
        y_pred = self.predict(x)
        if type(loss_function) == str:
            loss_function = getattr(objectives, loss_function)
        return loss_function(y_true, y_pred)

    def save(self, file_path):
        for i, layer in enumerate(self.layers):
            with h5py.File('layer_{0}.hdf5'.format(i),'w') as hfile:
                hfile.create_dataset('W', data=layer.W)
                hfile.create_dataset('b', data=layer.b)

    def restore(self, file_path):
        for i, layer in enumerate(self.layers):
            with h5py.File('layer_{0}.hdf5'.format(i),'r') as hfile:
                self.layers[i].W = hfile['W'][:]
                self.layers[i].b = hfile['b'][:]

    def output_error(self, target, output):
        return -(target - output)
