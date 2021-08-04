import numpy as np
import pandas as pd

# todo: Add a possibility to save the state of the network and add some tools to test against custom data.


class MLP:  # A standard multi layer perceptron.

    def __init__(self, nodes_input, nodes_output, nodes_hidden, layers, activation, activation_prime):

        self._layers = []

        self._input_layer = MLP.Layer(nodes_input, 0)

        self._layers.append(self._input_layer)

        self._hidden_layers = []

        self._activation = activation
        self._activation_derivative = activation_prime

        for i in range(layers):
            layer = MLP.Layer(nodes_hidden, i + 1, parent=self._layers[i])
            self._hidden_layers.append(layer)
            self._layers.append(layer)

        self._output_layer = MLP.Layer(nodes_output, layers + 1, parent=self._hidden_layers[-1])
        self._layers.append(self._output_layer)
        self._weight_sets = []
        self._bias_sets = []


    def forwardPass(self, data: np.ndarray) -> (np.ndarray, list):
        result_a = data
        result_sets = [result_a]

        for weights, bias in zip(self._weight_sets, self._bias_sets):

            result = np.dot(weights, result_a)
            result += np.reshape(bias, (10, 1))
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

                loss = np.dot(np.transpose(weights), loss) * self._activation_derivative(values[-(2+2*i)])

        return weight_correction[::-1], bias_correction[::-1]

    def train(self, train_data, label_data, iterations, convergence, learning_rate):

        for i in range(iterations):

            result, layer_values = self.forwardPass(train_data)
            loss = lossFunction(result, label_data)
            weight_corrections, bias_corrections = self.backPropagation(loss, layer_values)
            self.updateNetwork(weight_corrections, bias_corrections, learning_rate)
            predictions = self.getPredictions(train_data)
            accuracy = np.sum(predictions == label_data)/label_data.shape[0]
            if (i+1) % 50 == 0:
                print(f"Iteration: {i+1}")
                print(f"Training data accuracy: {accuracy*100}%")
            if accuracy >= convergence:
                print(f"Convergence criteria met at iteration {i} with accuracy of {accuracy*100}%, training stopped.")
                break

    def updateNetwork(self, weight_corrections, bias_corrections, learning_rate):

        for i, (weight_set, bias_set) in enumerate(zip(self._weight_sets, self._bias_sets)):
            weight_set -= learning_rate * weight_corrections[i]
            bias_set -= learning_rate * bias_corrections[i]

    def getPredictions(self, data):
        result, _ = self.forwardPass(data)
        return np.nonzero(np.transpose(result == np.amax(result, 0)))[1]

    def randomizeWeights(self):

        for i in range(len(self._layers) - 1):
            weights = np.random.rand(self._layers[i].nodes * self._layers[i + 1].nodes) - .5
            self._weight_sets.append(np.reshape(weights, (self._layers[i + 1].nodes, self._layers[i].nodes)))
            self._bias_sets.append(np.transpose(np.random.rand(self._layers[i + 1].nodes) - .5))

    def saveNetwork(self, filepath):

        rows = []

        for weight_set, bias_set in zip(self._weight_sets, self._bias_sets):
            rows.append([np.array(weight_set, dtype=object), np.array(bias_set, dtype=object)])

        data_frame = pd.DataFrame(rows, columns=['Weights', 'Bias'])
        data_frame.to_json(filepath)

    def loadNetwork(self, filepath):

        network_data = pd.read_json(filepath)

        if len(network_data['Weights']) != len(self._layers) - 1:  # Should also check for number of nodes, but
            # since this is purely educational script, I didn't bother.
            raise AttributeError(f"Loaded data does not match current model.")

        self._weight_sets = []
        self._bias_sets = []

        for weight_set, bias_set in zip(network_data['Weights'], network_data['Bias']):
            self._weight_sets.append(weight_set)
            self._bias_sets.append(bias_set)

    class Layer:  # This class is redundant, but I left it for clarity. Initially I wanted to store weights and
        # biases associated with each layer in this object, but I found it not very intuitive, as each weight set is
        # connected to two layers.
        def __init__(self, nodes, index, parent=None):

            self.nodes = nodes
            self._index = index
            self.parent = parent


def sigmoid(var: float) -> float:

    return 1/(1+np.exp(-var))


def sigmoidPrime(var: float) -> float:

    return sigmoid(var)*(1 - sigmoid(var))


def ReLU(var: np.ndarray) -> float:

    return np.maximum(0, var)


def ReLUPrime(var: np.ndarray):
    return var > 0


def softMax(vec: np.ndarray) -> float:

    return np.exp(vec) / sum(np.exp(vec))


def lossFunction(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # Initially I wanted to apply the cross entropy loss function but,
    # I couldn't be bothered to calculate the derivative.

    expected = []
    for i in range(max(labels.shape)):
        compare = np.array(np.zeros(10))
        compare[labels[i]] = 1
        expected.append(compare)

    expected = np.transpose(np.array(expected))

    error = data - expected

    return error
