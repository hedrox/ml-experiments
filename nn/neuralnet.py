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
    
    def fit(self, x, y_true, epochs=100000, learning_rate=0.01):
        #batches = len(x)/batch_size
        x_batch = x
        y_batch = y_true
        
        for epoch in range(epochs):
            for x_input, y_output in zip(x_batch,y_batch):
                #forward pass
                y_pred = self.predict(x_input)

                #backward pass
                # output layer error
                out_error = self.output_error(y_output, y_pred)
                delta = out_error * self.layers[-1].d_output(y_pred)
                delta = np.dot(delta, self.layers[-1].a.T)
                self.layers[-1].delta = delta

                # other layers error
                for layer in range(2,len(self.layers)):
                    last_layer_activation = self.layers[-layer-1].a
                    delta = self.layers[-layer].bwd_prop(delta, last_layer_activation, self.layers[-layer+1].W)
                
                #SGD
                for layer in self.layers[1:]:
                    layer.W += -learning_rate * layer.delta
            # for batch in range(batches):
            #     start_batch_idx = batch * batch_size
            #     end_batch_idx = start_batch_idx + batch_size

            #     x_batch = x[start_batch_idx:end_batch_idx]
            #     y_batch = y[start_batch_idx:end_batch_idx]

                
    def predict(self, x):
        input = x
        # ignore the input layer
        for layer in self.layers[1:]:
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

    # quadratic cost function
    def output_error(self, target, output):
        return -(target - output)
