import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Network:

    def __init__(self, nodes_input, nodes_output, nodes_hidden, layers, activation):

        self._layers = []

        self._input_layer = Network.Layer(nodes_input, 0)

        self._layers.append(self._input_layer)

        self._hidden_layers = []

        for i in range(layers):
            layer = Network.Layer(nodes_hidden, i+1, parent=self._layers[i])
            self._hidden_layers.append(layer)
            self._layers.append(layer)

        self._output_layer = Network.Layer(nodes_output, layers + 1, parent=self._hidden_layers[-1])
        self._layers.append(self._output_layer)

    def forwardPass(self, data: np.ndarray):

        return

    def calculateLoss(self, data: np.ndarray):

        return

    class Layer:

        def __init__(self, nodes, index, parent=None):

            self._nodes = nodes
            self._index = index
            if parent:
                self._parent = parent
            self.weights = np.transpose()


def sigmoid(var: float) -> float:

    return 1/(1+np.exp(-var))


def sigmoidPrime(var: float) -> float:

    return sigmoid(var)*(1 - sigmoid(var))


if __name__ == "main":

    train_data = pd.read_csv("data/mnist_train.csv")
    test_data = pd.read_csv("data/mnist_test.csv")

    train_data = np.transpose(np.array(train_data))
    test_data = np.transpose(np.array(test_data))

    learning_rate = .1
    nodesW1 = 10
    nodesW2 = 10

    ann = Network(785, 10, 10, 1, sigmoid)
