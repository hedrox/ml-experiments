import numpy as np

class Layer(object):
    def __init__(self, in_shape, out_shape, input_layer=False, W_stddev=1.0, activation='sigmoid', params={}):
        self.in_shape = in_shape
        self.out_shape = out_shape
        W_shape = (self.in_shape, self.out_shape)
        if not input_layer:
            self.W = np.random.normal(size=W_shape, scale=W_stddev) if not params else params['W']
            self.b = np.zeros(self.out_shape) if not params else params['b']

        if type(activation) == str:
            self.activation = globals().copy().get(activation)
        else:
            self.activation = activation
    
    def fwd_prop(self, layer_in):
        self.layer_in = layer_in
        self.a = self.activation(np.dot(layer_in, self.W.T) + self.b)
        return self.a
    
    def bwd_prop(self, delta, last_layer_activation, previous_layer_w):
        grad = d_sigmoid(self.a)
        delta = np.dot(previous_layer_w.T, delta) * grad
        self.delta = np.dot(delta, last_layer_activation.T)
        return self.delta

    @property
    def output(self):
        return self.a

    def d_output(self, activ):
        return self.activation(activ, deriv=True)
        


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def sigmoid(x, deriv=False):
    if deriv:
        return d_sigmoid(x)
    return 1.0/(1.0 + np.exp(-x))

def d_sigmoid(x):
    return x*(1.0-x)

def relu(x):
    return np.maximum(0.0,x)
