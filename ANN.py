import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt


class Network:

    def __init__(self, nodes_input, nodes_output, nodes_hidden, layers, activation):

        self._layers = []

        self._input_layer = Network.Layer(nodes_input, 0)

        self._layers.append(self._input_layer)

        self._hidden_layers = []

        self._activation = activation

        for i in range(layers):
            layer = Network.Layer(nodes_hidden, i+1, parent=self._layers[i])
            self._hidden_layers.append(layer)
            self._layers.append(layer)

        self._output_layer = Network.Layer(nodes_output, layers + 1, parent=self._hidden_layers[-1])
        self._layers.append(self._output_layer)

        self._weight_sets = []
        self._bias_sets = []

        for i in range(len(self._layers) - 1):

            weights = np.random.randn(self._layers[i].nodes * self._layers[i+1].nodes)
            self._weight_sets.append(np.reshape(weights, (self._layers[i+1].nodes, self._layers[i].nodes)))
            self._bias_sets.append(np.transpose(np.random.randn(self._layers[i+1].nodes)))

    def forwardPass(self, data: np.ndarray) -> (np.ndarray, list):
        result = data

        result_sets = []

        for weights, bias in zip(self._weight_sets, self._bias_sets):

            result = np.dot(weights, result)
            bias_matrix = np.transpose(np.reshape(np.tile(bias, result.shape[1]), (result.shape[1],result.shape[0])))
            result += bias_matrix
            result = self._activation(result)
            result_sets.append(result)

        return softMax(result), result_sets

    def calculateLoss(self, data: np.ndarray):

        return

    def backPropagation(self, result: np.ndarray, labels: np.ndarray):



        return

    class Layer:

        def __init__(self, nodes, index, parent=None):

            self.nodes = nodes
            self._index = index
            self.parent = parent
            #     self.weights = np.random.randn(self._parent._nodes * self._nodes)
            #     self.weights = np.reshape(self.weights,(self._nodes,self._parent._nodes ))
            # else:
            #     self.weights = np.random.randn(self._nodes)


def sigmoid(var: float) -> float:

    var = np.array(var, dtype=np.float128)  # Converting to float 128 to avoid overflow.
    return 1/(1+np.exp(-var))


def sigmoidPrime(var: float) -> float:

    return sigmoid(var)*(1 - sigmoid(var))


def ReLU(var: np.ndarray) -> float:

    return np.maximum(0, var)


def softMax(vec: np.ndarray) -> float:

    mod = np.exp(vec)
    return mod/np.sum(mod)

if __name__ == "__main__":

    train_data = pd.read_csv("data/mnist_train.csv")
    test_data = pd.read_csv("data/mnist_test.csv")

    train_data = np.transpose(np.array(train_data))
    test_data = np.transpose(np.array(test_data))

    learning_rate = .1
    examples = 40
    train = train_data[1:, :examples]
    tran_data_labels = train_data[0, :examples]

    ann = Network(784, 10, 10, 1, sigmoid)

    res, res2 = ann.forwardPass(train)
    print(len(res2))

    # for obj in ann._weight_sets:
    #
    #     print(obj.shape)

