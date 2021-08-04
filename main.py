import numpy as np
import pandas as pd
import ann

train_data = pd.read_csv("data/mnist_train.csv")
test_data = pd.read_csv("data/mnist_test.csv")

train_data = np.transpose(np.array(train_data))
test_data = np.transpose(np.array(test_data))

examples = 10000
train = train_data[1:, :examples] / 255.
train_data_labels = train_data[0, :examples]

learning_rate = 1.4

net = ann.MLP(784, 10, 10, 3, ann.sigmoid, ann.sigmoidPrime)

net.train(train, train_data_labels, 500, 0.65, learning_rate)