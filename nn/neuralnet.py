from typing import List, Union, Callable

import numpy as np
import h5py
import objectives
import layers


class NeuralNetwork:
    def __init__(self, layers: List[layers.Layer]):
        self.layers = layers

    def add(self, layers: Union[List[layers.Layer], layers.Layer]) -> None:
        if isinstance(layers, list):
            self.layers.extend(layers)
        else:
            self.layers.append(layers)

    def fit(self, x: np.ndarray, y_true: np.ndarray,
            epochs: int = 100000, learning_rate: float = 0.01) -> None:
        x_batch = x
        y_batch = y_true

        for _ in range(epochs):
            for x_input, y_output in zip(x_batch,y_batch):
                # forward pass
                y_pred = self.predict(x_input)

                # backward pass
                # output layer error
                out_error = NeuralNetwork.output_error(y_output, y_pred)
                delta = out_error * self.layers[-1].d_output(y_pred)
                self.output_bwd_prop(delta)

                # hidden layers error
                for layer in reversed(range(1,len(self.layers)-1)):
                    layer_activation = x_input if layer == 1 else self.layers[layer-1].activ
                    delta = self.layers[layer].bwd_prop(delta, layer_activation,
                                                        self.layers[layer+1].W)

                self.SGD(learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # ignore the input layer
        for layer in self.layers[1:]:
            x = layer.fwd_prop(x)
        return x

    def loss(self, x: np.ndarray, y_true: np.ndarray, loss_function: Callable) -> np.float64:
        y_pred = self.predict(x)
        if isinstance(loss_function, str):
            loss_function = getattr(objectives, loss_function)
        return loss_function(y_true, y_pred)

    def SGD(self, learning_rate: float) -> None:
        # ignore the input layer
        for layer in self.layers[1:]:
            layer.W -= learning_rate * layer.dW

    def output_bwd_prop(self, delta: np.ndarray) -> np.ndarray:
        dw = []
        for d in delta:
            for a in self.layers[-2].activ:
                dw.append(d*a)
        dW = np.array(dw).reshape(self.layers[-1].W.shape)
        self.layers[-1].dW = dW
        return dW

    def save(self) -> None:
        for i, layer in enumerate(self.layers):
            with h5py.File('layer_{0}.hdf5'.format(i),'w') as hfile:
                hfile.create_dataset('W', data=layer.W)
                hfile.create_dataset('b', data=layer.b)

    def restore(self) -> None:
        for i, _ in enumerate(self.layers):
            with h5py.File('layer_{0}.hdf5'.format(i),'r') as hfile:
                self.layers[i].W = hfile['W'][:]
                self.layers[i].b = hfile['b'][:]

    @staticmethod
    def output_error(target: np.ndarray, output: np.ndarray) -> np.ndarray:
        # quadratic cost function
        return -(target - output)
