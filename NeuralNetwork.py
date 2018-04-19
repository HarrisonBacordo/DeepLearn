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
        self.hidden = list()
        for i, units in enumerate(structure):
            # output layer
            if i == len(structure) - 1:
                self.output = Layer("output", units, inputs=structure[i-2], activation=activation)
            # hidden layer
            elif i != 0:
                self.hidden.append(Layer("hidden", units, inputs=structure[i - 1], activation=activation))

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
        # featurecount = 0
        # for _ in range(epochs):
        #     batchcosts = list()
        #     for _ in range(batch_size):
        #         guess = self.feed_forward(features_list[featurecount])
        #         if max(guess) != labels[featurecount].nonzero():
        #             batchcosts.append(self.calculate_cost(guess, labels[featurecount]))
        #         featurecount += 1
        #     if batchcosts:
        #         batchcosts = np.asarray(batchcosts)
        #         avgcost = np.mean(batchcosts, axis=0)
        #         self.compute_gradient(avgcost)
        self.lr = learning_rate
        featurecount = 0
        for _ in range(epochs):
            for _ in range(batch_size):
                guess, states = self.feed_forward(features_list[featurecount])
                cost = np.array(self.calculate_cost(guess, labels[featurecount]))
                self.compute_gradient(cost, states)
                featurecount += 1

    def feed_forward(self, features):
        """
        initiates the feed forward algorithm for the given features
        :param features: features to feed through the neural network
        :return: the resulting guess (in the form of a vector)
        """
        states = list()
        state = features
        # hidden
        for i, hlayer in enumerate(self.hidden):
            state = hlayer.feed_forward(state)
            states.append(state)
        # output
        state = self.output.feed_forward(state)
        states.append(state)
        return state, states

    @staticmethod
    def calculate_cost(guess, label):
        """
        Calculates the cost of the guess given the label
        :param guess: NN's guess
        :param label: actual answer
        :return: cost of guess given the label
        """
        costs = list()
        for i in range(len(guess)):
            costs.append(pow(guess[i] - label[i], 2))
        return costs

    def compute_gradient(self, costs, states: np.ndarray):
        """
        Implements SGD on the costs that are passed in
        :param costs: avg costs of this mini batch
        :param states: current state of the neural network
        :return: TODO Unsure of this right now
        """
        # calculate output gradient
        dsig = np.vectorize(derivsig)
        hstate = np.array(states[:-1])
        ostate = np.array(states[-1])
        ogradient = dsig(ostate)
        ogradient = np.multiply(ogradient, costs)
        ogradient = np.multiply(ogradient, self.lr)
        # calculate output deltas
        hidden_t = np.transpose(hstate[-1])
        hidden_ho_deltas = np.multiply(ogradient, hidden_t)
        self.output.applydeltas(hidden_ho_deltas)

        for i in reversed(range(len(hstate))):
            # calculate hidden gradient
            if i == len(hstate):
                errors = self.output.compute_error(costs)
            else:
                errors = self.hidden[i].compute_error()
            hgradient = dsig(hstate[i])
            hgradient = np.multiply(hgradient, errors)
            hgradient = np.multiply(hgradient, self.lr)
            # calculate hidden deltas
            hidden_t = np.transpose(hstate[i-1])
            weight_hh_deltas = np.multiply(hgradient, hidden_t)
            self.hidden[i].applydeltas(weight_hh_deltas)

        # for hidden in hstate
        for hidden in reversed(self.hidden):

            hidden.compute_error(errors)


class Layer:
    def __init__(self, _type, units, inputs, activation=None):
        """
        Creates a layer within a neural network
        :param _type: Type of layer, could be either input, hidden, or output
        :param units: number of perceptrons in this layer
        :param inputs: the input that the perceptrons in this layer will be receiving
        :param activation: the activation function used for perceptrons in this layer
        """
        self._type = _type
        self.activation = activation
        self.inputs = inputs
        self.units = units
        self.weights = np.random.rand(units, inputs)
        # print(_type, ":\n", self.weights, "\n\n")

    def feed_forward(self, inputs):
        """
        feeds forward the inputs to all of this layers perceptrons.
        :param inputs: the inputs to feed to the perceptrons of this layer
        :return: a vector containing the results of each perceptron's computations
        """
        _sum = self.weights.dot(inputs)
        sig = np.vectorize(sigmoid)

        if self.activation == "sigmoid":
            return sig(_sum)
        return None

    def compute_error(self, costs):
        """
        computes error of the given layer, given the costs
        :param costs: error of the layer ahead of this layer
        :return:
        """
        errors = np.zeros(self.inputs)
        flippedweights = np.transpose(self.weights)
        for i, weights in enumerate(flippedweights):
            errors[i] = np.dot(weights, costs)
        print("NEW\n", errors, "\n\n")
        return errors

    def applydeltas(self, deltas):
        self.weights = np.add(self.weights, deltas)


def sigmoid(x):
    """
    implements sigmoid on num
    :param x: number to implement sigmoid on
    :return: result from sigmoid calculation
    """
    return x / (1 + abs(x))


def derivsig(y):
    """
    implements derivative of sigmoid
    :param y: number to implement derivsig on
    :return: result from derivsig calculation
    """
    return y * (1-y)


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


x = NeuralNetwork([5, 10, 10, 5], "sigmoid")
data = list()
answers = list()
for _ in range(200):
    data.append(np.random.random_integers(0, 100, 5))
    answers.append(random.randint(0, 5))
answers = one_hot(answers)

x.train(data, answers, 10)
