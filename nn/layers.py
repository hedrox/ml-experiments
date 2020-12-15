import numpy as np


class Layer:
    def __init__(self, in_shape, out_shape, input_layer=False,
                 W_stddev=1.0, activation='sigmoid', params=None):
        self.in_shape = in_shape
        self.out_shape = out_shape
        W_shape = (self.in_shape, self.out_shape)
        if not input_layer:
            self.W = np.random.normal(size=W_shape, scale=W_stddev) if not params else params['W']
            self.b = np.zeros(self.out_shape) if not params else params['b']

        if isinstance(activation, str):
            self.activation = globals().copy().get(activation)
        else:
            self.activation = activation

    def fwd_prop(self, layer_in):
        self.activ = self.activation(np.dot(layer_in, self.W.T) + self.b)
        return self.activ

    def bwd_prop(self, delta, layer_activation, layer_w):
        out_delta = np.dot(delta, layer_w)
        grad = self.activation(self.activ, deriv=True)
        delta = out_delta * grad
        dw = []
        for d in delta:
            for a in layer_activation:
                dw.append(d*a)
        dW = np.array(dw).reshape(layer_w.shape)
        self.dW = dW
        return delta

    @property
    def output(self):
        return self.activ

    def d_output(self, activ):
        return self.activation(activ, deriv=True)


class ConvLayer(Layer):
    def __init__(self, in_shape, n_filter, filter_shape, strides,
                 W_stddev=1.0, padding_mode='same', activation='linear'):
        self.in_shape = in_shape
        self.n_filter = n_filter
        self.filter_shape = filter_shape
        self.strides = strides
        self.padding_mode = padding_mode
        self.activation = activation
        W_shape = (self.in_shape, self.n_filter) + self.filter_shape
        self.W = np.random.normal(scale=W_stddev, size=W_shape)
        self.b = np.zeros(self.n_filter)


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def sigmoid(x, deriv=False):
    if deriv:
        return d_sigmoid(x)
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(a):
    return a*(1.0-a)

def relu(x):
    return np.maximum(0.0,x)
