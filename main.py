import ann
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

test_data = pd.read_csv("data/mnist_test.csv")
test_data = np.transpose(np.array(test_data))

data = test_data[1:, :] / 255.
labels = test_data[0, :]

examples = 50000

net = ann.MLP(784, 10, 10, 1, ann.sigmoid, ann.sigmoidPrime)

net.loadNetwork('model data/net.json')

chosen_image = np.random.randint(0, labels.shape[0] - 1)
im_size = int(np.sqrt(784))

predicted = net.getPredictions(np.reshape(data[:, chosen_image], (784, 1)))
predictions = net.getPredictions(data)
accuracy = np.sum(predictions == labels)/labels.shape[0]

print(f'Expected number: {labels[chosen_image]}')
print(f'Predicted number: {predicted}')
print(f'Accuracy across the entire testing dataset: {accuracy * 100}%')

plt.imshow(np.reshape(data[:, chosen_image], (im_size, im_size)), cmap='gray')
plt.show()
