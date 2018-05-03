import FullyConnected as nn
import numpy as np


net = nn.FullyConnected([2, 12, 2], "sigmoid")
data = np.zeros((100000, 2))
answers = list()
for n in range(100000):
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
answers = nn.one_hot(answers)

net.train(data, answers, 1000, batch_size=100, momentum=0.8, learning_rate=5)
