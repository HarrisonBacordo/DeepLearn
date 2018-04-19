import numpy as np


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
        self.structure = list()
        self.hidden = list()
        for i, units in enumerate(structure):
            if i == 0:
                self.input = units
            elif i == len(structure):
                self.op = units
            self.structure.append(np.random.uniform(-1, 1, ()))
            # output layer
            if i == len(structure) - 1:
                self.output = Layer("output", units, inputs=structure[i - 1], activation=activation)
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
        self.lr = learning_rate
        featurecount = 0  # for iterating through features
        for epoch in range(epochs):
            correct = 0
            for _ in range(batch_size):
                # store the nn's guess and a list of each layer's values
                guess, states = self.feed_forward(features_list[featurecount])
                # m = np.argmax(guess)
                # n = list(labels[featurecount]).index(1)
                # print("GUESS: ", m, "\nTARGET: ", n, "\n\n")
                # check if nn guessed correct
                if np.argmax(guess) == list(labels[featurecount]).index(1):
                    correct += 1
                # calculate cost and compute the gradient
                cost = np.array(self.calculate_cost(guess, labels[featurecount]))
                self.compute_gradient(cost, states)
                featurecount += 1
            print("EPOCH ", epoch, ": ", correct, "/", batch_size)

    def feed_forward(self, features):
        """
        feeds the features forward through the NN
        :param features: features to feed through the neural network
        :return: the resulting guess (in the form of a vector), along with the state of the NN
        """
        states = list()  # stores the states of each layer
        state = features  # starts with features; they are input layer's values being passed to hidden
        states.append(state)
        # feed forward through the hidden layers.
        for hlayer in self.hidden:
            state = hlayer.feed_forward(state)
            states.append(state)
        # feed forward through the output. This state is the nn's guess
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
        # TODO maybe move this out of a function?
        return np.subtract(guess, label)

    def compute_gradient(self, costs, states: np.ndarray):
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
        istate = np.array(states[0])
        hstate = np.array(states[1:-1])
        ostate = np.array(states[-1])
        # get the gradient of the output state.
        # multiply it by costs vector and learning rate
        ogradient = dsig(ostate)
        ogradient = np.multiply(ogradient, costs)
        ogradient = np.multiply(ogradient, self.lr)
        # transpose the the hlayer connected to the outlayer.
        hidden_t = np.transpose(np.atleast_2d(hstate[-1]))
        # multiply the output gradient by the transposed hidden layer.
        # This is the change to be applied to the output weights.
        hidden_ho_deltas = np.multiply(hidden_t, ogradient)
        self.output.applydeltas(hidden_ho_deltas)

        #     #     #     #     HIDDEN LAYER     #     #     #     #
        hcost = costs  # stores the cost of the -> layer
        for i in reversed(range(len(hstate))):
            # calculate hidden gradient

            # gets the errors for this layer based on the -> layer error
            if i == len(hstate) - 1:   # output layer
                hcost = self.output.compute_error(costs)
            else:
                hcost = self.hidden[i+1].compute_error(hcost)
            # get the gradients of this layer
            # multiply this layer's gradients by the cost of the -> layer.
            # multiply the above by the learning rate scalar
            hgradients = dsig(hstate[i])
            hgradients = np.multiply(hgradients, hcost)
            hgradients = np.multiply(hgradients, self.lr)
            # transpose the <- layer's state

            if i == 0:
                hidden_t = np.transpose(np.atleast_2d(istate))
            else:
                hidden_t = np.transpose(np.atleast_2d(hstate[i - 1]))

            # multiply this layer's gradients by the transposed <- layer
            # This is the change to be applied to this layer's weights.
            weight_hh_deltas = np.multiply(hgradients, hidden_t)
            self.hidden[i-1].applydeltas(weight_hh_deltas)


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
        self.weights = np.random.uniform(-1, 1, (units, inputs))
        # print(_type, ":\n", self.weights, "\n\n")

    def feed_forward(self, inputs):
        """
        feeds forward the inputs to all of this layers perceptrons.
        :param inputs: the inputs to feed to the perceptrons of this layer
        :return: a vector containing the results of each perceptron's computations
        """
        # dot product == x1w1 + x2w2 + ... + xiwi
        # does it for each unit in the layer
        _sum = self.weights.dot(inputs)

        # map sigmoid each value in _sum matrix
        sig = np.vectorize(sigmoid)   # vectorize sigmoid function
        if self.activation == "sigmoid":
            return sig(_sum)
        return None

    def compute_error(self, costs):
        """
        computes error of the given layer, given the costs of the -> layer.
        Transposes this layer's weight matrix and finds the dot product for
        each row in weights_t and the costs vector
        :param costs: error of the -> layer
        :return:
        """

        # transpose weights matrix so
        weights_t = np.transpose(self.weights)
        return np.dot(weights_t, costs)

    def applydeltas(self, deltas):
        self.weights = np.add(self.weights, deltas)


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
data = list()
answers = list()
for _ in range(100000):
    i = np.random.randint(0, 2, 2)
    data.append(i)
    if 1 in i and 0 in i:
        answers.append(1)
    else:
        answers.append(0)
answers = one_hot(answers)

x.train(data, answers, 1000, batch_size=100)
