import NeuralNetwork as nn
import numpy as np


nn = nn.NeuralNetwork([2, 2, 1], "sigmoid")
data = np.zeros((1000, 2))
answers = list()
for n in range(1000):
    rand = np.random.randint(0, 2, 2)
    data[n] = rand
    if 1 in rand and 0 in rand:
        answers.append(1)
    else:
        answers.append(0)
    # rand = np.random.randint(0, 2)
    # data[n] = [rand]
    # if rand == 1:
    #     labels.append(1)
    # else:
    #     labels.append(0)
# labels = one_hot(labels)

nn.train(data, answers, 1000, batch_size=100)