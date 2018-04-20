import numpy as np
import random


# TODO Implement SGD
class NeuralNetwork:
    def __init__(self, structure, activation):
        """
        Creates skeleton of neural network
        :param structure: a list that gives the number of layers (length of list) and the number of units in those
        layers (numbers in list).
        :param activation: type of activation function to use throughout neural network
        """
        self.lr = None
        self.structure = structure
        self.bias = list()
        self.nn = list()
        for i in range(len(self.structure) - 1):
            self.nn.append(np.random.rand(self.structure[i+1], self.structure[i]))
            self.bias.append(np.ones(self.structure[i+1])[:, None])
        print()

    def train(self, features_list, labels, epochs, batch_size=20, learning_rate=0.1):
        """
        Trains the neural network on the given data
        :param features_list: the input data for the neural network
        :param labels: the correct answers for each of the features
        :param epochs: amount of batch_sizes to do before halting training
        :param batch_size: amount of guesses to do before each backprop
        :param learning_rate: the rate at which to attempt to optimize the model
        :return:
        """
        self.lr = learning_rate
        for i, features in enumerate(features_list):
            # store the nn's guess and a list of each layer's values
            guess, states = self.feed_forward(features)
            # print("INPUT: ", features, "OUTPUT: ", guess)
            # check if nn guessed correct
            # if np.argmax(guess) == list(labels[featurecount]).index(1):
            #     correct += 1
            # calculate cost and compute the gradient
            cost = np.array(np.subtract(guess, labels[i]))
            self.compute_gradient(np.atleast_2d(cost), states)
            # print("EPOCH ", epoch, ": ", correct, "/", batch_size)

    def feed_forward(self, features):
        """
        feeds the features forward through the NN
        :param features: features to feed through the neural network
        :return: the resulting guess (in the form of a vector), along with the state of the NN
        """
        sigm = np.vectorize(sigmoid)
        states = list()
        state = np.array(features)[:, None]
        states.append(state)
        for i, weights in enumerate(self.nn):
            if i == 0:
                print(weights, "\n")
            state = np.dot(weights, state)
            state = np.add(state, self.bias[i])
            state = sigm(state)
            states.append(state)
        return state, states

    def compute_gradient(self, costs, states):
        """
        Implements SGD on the costs that are passed in
        :param costs: avg costs of this mini batch
        :param states: current state of the neural network
        :return: TODO Unsure of this right now
        """

        #     #     #     #      OUTPUT LAYER     #     #     #     #

        # calculate output gradient
        dsig = np.vectorize(derivsig)  # vectorize the derivsig function
        # grab hidden layers, and output layer
        istate = states[0]
        hstate = states[1:-1]
        ostate = states[-1]
        # get the gradient of the output state.
        # multiply it by costs vector and learning rate
        ogradient = dsig(ostate)
        ogradient = np.multiply(costs, ogradient)
        ogradient = np.multiply(ogradient, self.lr)
        # transpose the the hlayer connected to the outlayer.
        hidden_t = np.transpose(hstate[-1])
        # multiply the output gradient by the transposed hidden layer.
        # This is the change to be applied to the output weights.
        weight_ho_deltas = np.dot(hidden_t, ogradient)
        self.bias[-1] = np.add(self.bias[-1], ogradient)
        self.nn[-1] = np.add(self.nn[-1], weight_ho_deltas)

        #     #     #     #     HIDDEN LAYER     #     #     #     #
        hcost = costs  # stores the cost of the -> layer
        hcount = len(hstate)
        for i in reversed(range(len(self.nn)-1)):
            # calculate hidden gradient

            # gets the errors for this layer based on the -> layer errort layer
            who_t = np.transpose(self.nn[i+1])
            hcost = who_t.dot(hcost)
            # get the gradients of this layer
            # multiply this layer's gradients by the cost of the -> layer.
            # multiply the above by the learning rate scalar
            hgradients = dsig(hstate[i])
            hgradients = np.multiply(hgradients, hcost)
            hgradients = np.multiply(hgradients, self.lr)
            # transpose the <- layer's state

            if i == 0:
                hidden_t = np.transpose(istate)
            else:
                hidden_t = np.transpose(hstate[hcount-1])

            # multiply this layer's gradients by the transposed <- layer
            # This is the change to be applied to this layer's weights.
            weight_hh_deltas = np.dot(hidden_t, hgradients)
            self.bias[i] = np.add(self.bias[i], hgradients)
            self.nn[i] = np.add(self.nn[i], weight_hh_deltas)
            hcount -= 1

    @staticmethod
    def compute_error(costs, weights):
        who_t = np.transpose(weights)
        return np.dot(who_t, costs)


def sigmoid(x):
    """
    implements sigmoid on num
    :param x: number to implement sigmoid on
    :return: result from sigmoid calculation
    """
    return 1 / (1 + np.exp(-x))




def derivsig(y):
    """
    implements derivative of sigmoid
    :param y: number to implement derivsig on
    :return: result from derivsig calculation
    """
    return y * (1 - y)


def one_hot(labels):
    """
    Converts the passed in labels into one hot format
    :param labels: labels to be converted
    :return: nparray of labels in one hot format
    """
    # get number of unique labels
    unique = list(set(labels))
    numclasses = len(unique)
    one_hots = list()
    for label in labels:
        onehot = np.zeros(numclasses)
        onehot[unique.index(label)] = 1
        one_hots.append(onehot)
    return one_hots


x = NeuralNetwork([2, 2, 2], "sigmoid")
data = np.zeros((100000, 2))
answers = list()
for n in range(100000):
    i = np.random.randint(0, 2, 2)
    data[n] = i
    if 1 in i and 0 in i:
        answers.append(1)
    else:
        answers.append(0)
answers = one_hot(answers)

print(sigmoid(0.99998 * 0.85637 + 1))

x.train(data, answers, 1000, batch_size=100)

state, _ = x.feed_forward([0, 1])
print("OUTPUT: ", state)
state, _ = x.feed_forward([1, 0])
print("OUTPUT: ", state)


state, _ = x.feed_forward([0, 0])
print("OUTPUT: ", state)


state, _ = x.feed_forward([1, 1])
print("OUTPUT: ", state)

