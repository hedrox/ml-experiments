import numpy as np

class Layer(object):
    def __init__(self, in_shape, out_shape, W_stddev=1.0, activation='relu'):
        self.in_shape = in_shape
        self.out_shape = out_shape
        W_shape = (self.in_shape[0], self.out_shape)
        self.W = np.random.normal(size=W_shape, scale=W_stddev)
        self.b = np.zeros(self.out_shape)

        if type(activation) == str:
            self.activation = globals().copy().get(activation)
        else:
            self.activation = activation
    
    def fwd_prop(self, layer_in):
        self.layer_in = layer_in
        return self.activation(np.dot(layer_in, self.W) + self.b)
    
    def bwd_prop(self, layer_out):
        grad = d_sigmoid(layer_out)
        delta = -(target - layer_out) * grad * self.layer_in
        return delta

    @property
    def output(self):
        return self.activation(np.dot(self.layer_in, self.W) + self.b)
        


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1.0-sig)

def relu(x):
    return np.maximum(0.0,x)
