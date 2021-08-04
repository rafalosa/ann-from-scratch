import pandas as pd
import ann
import numpy as np

train_data = pd.read_csv("data/mnist_train.csv")
test_data = pd.read_csv("data/mnist_test.csv")

train_data = np.transpose(np.array(train_data))
test_data = np.transpose(np.array(test_data))

examples = 50000
train = train_data[1:, :examples] / 255.
train_data_labels = train_data[0, :examples]

learning_rate = .4

net = ann.MLP(784, 10, 10, 1, ann.sigmoid, ann.sigmoidPrime)

net.randomizeWeights()

net.train(train, train_data_labels, 1000, 0.9, learning_rate)
net.saveNetwork('model data/net.json')
