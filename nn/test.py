import unittest
import numpy as np
import neuralnet
import layers

class Network(unittest.TestCase):
    def setUp(self):
        params_layer2 = {'W': np.array([[-0.5, 0.7],[1.3, -1]]), 'b': np.array([0.5, 0.7])}
        params_layer3 = {'W': np.array([[0.5, 0.3],[0.2, 1]]), 'b': np.array([1, 0.5])}
        nn_layers = [layers.Layer(2, 2, input_layer=True), layers.Layer(2,2,params=params_layer2), layers.Layer(2,2, params=params_layer3)]
        self.nn = neuralnet.NeuralNetwork(nn_layers)

    def test_feedforward(self):
        X = np.array([0,1])
        y_pred = self.nn.predict(X)
        self.assertEqual('{0:.3f}'.format(y_pred[0]), '0.819')
        self.assertEqual('{0:.3f}'.format(y_pred[1]), '0.746')


# layers = [layers.Layer(2,2), layers.Layer(2,2), layers.Layer(2,1)]

# X = np.array([[0,0],
#               [0,1],
#               [1,0],
#               [1,1]])

# Y = np.array([0,1,1,0])
# nn = neuralnet.NeuralNetwork(layers)
# nn.fit(X, Y)
# for e in X:
#     print e, nn.predict(e)