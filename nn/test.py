import numpy as np
import neuralnet
import layers

layers = [layers.Layer(2,2), layers.Layer(2,2), layers.Layer(2,1)]

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([0,1,1,0])
nn = neuralnet.NeuralNetwork(layers)
nn.fit(X, Y)
for e in X:
    print e, nn.predict(e)