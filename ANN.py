import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt


class Network:

    def __init__(self, nodes_input, nodes_output, nodes_hidden, layers, activation,activation_prime):

        self._layers = []

        self._input_layer = Network.Layer(nodes_input, 0)

        self._layers.append(self._input_layer)

        self._hidden_layers = []

        self._activation = activation
        self._activation_derivative = activation_prime

        for i in range(layers):
            layer = Network.Layer(nodes_hidden, i+1, parent=self._layers[i])
            self._hidden_layers.append(layer)
            self._layers.append(layer)

        self._output_layer = Network.Layer(nodes_output, layers + 1, parent=self._hidden_layers[-1])
        self._layers.append(self._output_layer)

        self._weight_sets = []
        self._bias_sets = []

        for i in range(len(self._layers) - 1):

            weights = np.random.rand(self._layers[i].nodes * self._layers[i+1].nodes) - .5
            self._weight_sets.append(np.reshape(weights, (self._layers[i+1].nodes, self._layers[i].nodes)))
            self._bias_sets.append(np.transpose(np.random.randn(self._layers[i+1].nodes)))

    def forwardPass(self, data: np.ndarray) -> (np.ndarray, list):
        result = data
        result_sets = []
        result_sets.append(result)

        for weights, bias in zip(self._weight_sets, self._bias_sets):

            result = np.dot(weights, result)
            bias_matrix = np.transpose(np.reshape(np.tile(bias, result.shape[1]), (result.shape[1],result.shape[0])))
            result += bias_matrix
            result_a = self._activation(result)
            result_sets.append(result)
            result_sets.append(result_a)

        return softMax(result), result_sets

    def backPropagation(self, loss: np.ndarray, values: list):

        weight_correction = []
        bias_correction = []
        values = values[:-2]

        for i, weights in enumerate(self._weight_sets[::-1]):
            dw = 1/loss.shape[1] * np.dot(loss, np.transpose(values[-(1+2*i)]))
            db = 1/loss.shape[1] * np.sum(loss, 1)
            weight_correction.append(dw)
            bias_correction.append(db)
            if i != len(self._weight_sets)-1:
                loss = np.dot(np.transpose(weights),loss) * self._activation_derivative(-(2+2*i))

        return weight_correction[::-1], bias_correction[::-1]

    def train(self,train_data, label_data, iterations, learning_rate):

        for i in range(iterations):

            result, layer_values = ann.forwardPass(train_data)
            loss = lossFunction(result, label_data)
            weight_corrections, bias_corrections = ann.backPropagation(loss, layer_values)
            self.updateNetwork(weight_corrections, bias_corrections, learning_rate)
            if i % 50 == 0:
                predictions = self.getPredictions(train_data)
                accuracy = np.sum(predictions == label_data)/label_data.shape[0]
                print(f"Iteration: {i}")
                print(f"Training data accuracy: {accuracy}")

    def updateNetwork(self, weight_corrections, bias_corrections, learning_rate):

        for i, (weight_set, bias_set) in enumerate(zip(self._weight_sets, self._bias_sets)):
            weight_set -= learning_rate * weight_corrections[i]
            bias_set -= learning_rate * bias_corrections[i]

    def getPredictions(self,data):
        result, _ = self.forwardPass(data)
        print(result.shape)
        return np.nonzero(np.transpose(result == np.amax(result, 0)))


    class Layer:

        def __init__(self, nodes, index, parent=None):

            self.nodes = nodes
            self._index = index
            self.parent = parent


def sigmoid(var: float) -> float:

    var = np.array(var, dtype=np.float128)  # Converting to float 128 to avoid overflow.
    return 1/(1+np.exp(-var))


def sigmoidPrime(var: float) -> float:

    return sigmoid(var)*(1 - sigmoid(var))


def ReLU(var: np.ndarray) -> float:

    var = np.array(var, dtype=np.float128)
    return np.maximum(0, var)


def ReLUPrime(var: np.ndarray):
    return var > 0


def softMax(vec: np.ndarray) -> float:

    vec = np.array(vec, dtype=np.float128)
    return np.exp(vec)/np.sum(np.exp(vec))


def lossFunction(data: np.ndarray, labels: np.ndarray) -> np.ndarray:

    expected = []
    for i in range(max(labels.shape)):
        compare = np.array(np.zeros(10))
        compare[labels[i]] = 1
        expected.append(compare)

    expected = np.transpose(np.array(expected))

    return data - expected


if __name__ == "__main__":

    train_data = pd.read_csv("data/mnist_train.csv")
    test_data = pd.read_csv("data/mnist_test.csv")

    train_data = np.transpose(np.array(train_data))
    test_data = np.transpose(np.array(test_data))

    learning_rate = 1.1
    examples = 2000
    train = train_data[1:, :examples]/255.
    train_data_labels = train_data[0, :examples]

    ann = Network(784, 10, 10, 1, ReLU, ReLUPrime)

    ann.train(train, train_data_labels, 500, .1)


    # for obj in ann._weight_sets:
    #
    #     print(obj.shape)

