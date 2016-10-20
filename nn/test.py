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

    def test_backpropagation(self):
        X = np.array([0,1])
        y_output = np.array([1,0])
        y_pred = self.nn.predict(X)

        out_error = self.nn.output_error(y_output, y_pred)
        self.assertEqual('{0:.3f}'.format(out_error[0]), '-0.181')
        self.assertEqual('{0:.3f}'.format(out_error[1]), '0.746')

        delta = out_error * self.nn.layers[-1].d_output(y_pred)
        self.assertEqual('{0:.3f}'.format(delta[0]), '-0.027')
        self.assertEqual('{0:.3f}'.format(delta[1]), '0.141')

        dw = []
        for delt in delta:
            for activ in self.nn.layers[-2].a:
                dw.append(delt*activ)
        
        output_dW = np.array(dw).reshape(self.nn.layers[-1].W.shape)
        self.assertEqual('{0:.3f}'.format(output_dW[0][0]), '-0.021')
        self.assertEqual('{0:.3f}'.format(output_dW[0][1]), '-0.011')
        self.assertEqual('{0:.3f}'.format(output_dW[1][0]), '0.109')
        self.assertEqual('{0:.3f}'.format(output_dW[1][1]), '0.060')
        self.nn.layers[-1].dW = output_dW

        # hidden layers error
        for layer in reversed(range(1,len(self.nn.layers)-1)):
            try:
                layer_activation = self.nn.layers[layer-1].a
            except AttributeError:
                layer_activation = X
            delta = self.nn.layers[layer].bwd_prop(delta, layer_activation, self.nn.layers[layer+1].W)
        
        hidden_dW = self.nn.layers[-2].dW
        self.assertEqual('{0:.3f}'.format(hidden_dW[0][0]), '0.000')
        self.assertEqual('{0:.3f}'.format(hidden_dW[0][1]), '0.003')
        self.assertEqual('{0:.3f}'.format(hidden_dW[1][0]), '0.000')
        self.assertEqual('{0:.3f}'.format(hidden_dW[1][1]), '0.033')
    

# layers = [layers.Layer(2,2,input_layer=True), layers.Layer(2,2), layers.Layer(2,2)]

# X = np.array([[0,0],
#               [0,1],
#               [1,0],
#               [1,1]])

# Y = np.array([0,1,1,0])

# X = np.array([[0,1],[1,0]])
# Y = np.array([[1,0],[1,1]])

# nn = neuralnet.NeuralNetwork(layers)
# nn.fit(X, Y)
# for e in X:
#     print e, nn.predict(e)