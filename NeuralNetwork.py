import numpy as np


# TODO function to convert labels to onehot format. Makes it easier to calculate cost. Implement SGD
class NeuralNetwork:
    def __init__(self, structure, activation, zeroes=True):
        """
        Creates skeleton of neural network
        :param structure: a list that gives the number of layers (length of list) and the number of units in those
        layers (numbers in list).
        :param activation: type of activation function to use throughout neural network
        :param zeroes: Whether to initialize all variables in the network to zero or not
        """
        self.__hidden = list()
        for i, units in enumerate(structure):
            # input layer
            if i == 0:
                self.__input = Layer("input", units, None, activation=activation)
            # output layer
            elif i == len(structure) - 1:
                self.__output = Layer("output", units, inputs=self.__hidden[-1], activation=activation)
            # hidden layer
            else:
                # first hidden layer
                if not self.__hidden:
                    self.__hidden.append(Layer("hidden", units, inputs=self.__input, activation=activation))
                # non-first hidden layer
                else:
                    self.__hidden.append(Layer("hidden", units, inputs=self.__hidden[i - 2], activation=activation))

    def train(self, features_list, labels, epochs, batch_size=20, learning_rate=0.001):
        """
        Trains the neural network on the given data
        :param features_list: the input data for the neural network
        :param labels: the correct answers for each of the features
        :param epochs: amount of batch_sizes to do before halting training
        :param batch_size: amount of guesses to do before each backprop
        :param learning_rate: the rate at which to attempt to optimize the model
        :return:
        """
        featurecount = 0
        for _ in range(epochs):
            batchcosts = list()
            for _ in range(batch_size):
                guess = self.__feedforward(features_list[featurecount])
                if max(guess) != max(labels[featurecount]):
                    batchcosts.append(self.__calculate_cost(guess, labels[featurecount]))
                featurecount += 1
            if batchcosts:
                batchcosts = np.asarray(batchcosts)
                avgcost = np.mean(batchcosts, axis=0)
                self.__computegradient(avgcost)

    def __feedforward(self, features):
        """
        initiates the feed forward algorithm for the given features
        :param features: features to feed through the neural network
        :return: the resulting guess (in the form of a vector)
        """
        # input
        results = self.__input.feedforward(features)
        # hidden
        for layer in self.__hidden:
            results = layer.feedforward(results)
        # output
        results = self.__output.feedforward(results)
        return results

    @staticmethod
    def __calculate_cost(guess, label):
        """
        Calculates the cost of the guess given the label
        :param guess: NN's guess
        :param label: actual answer
        :return: cost of guess given the label
        """
        # TODO: WONT WORK, SINCE LABEL IS CURRENTLY NOT IN OHF
        costs = list()
        for i in range(len(guess)):
            costs.append(pow(guess[i] - label[i], 2))
        return costs

    def __computegradient(self, costs):
        """
        Implements SGD on the costs that are passed in
        :param costs: avg costs of this mini batch
        :return: TODO Unsure of this right now
        """
        return None


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
        self.units = self.__createlayer(units, activation, inputs)

    @staticmethod
    def __createlayer(units, activation, inputs):
        """
        Creates a layer with the given number of units, using the given activation function
        :param units: number of units for this layer
        :param activation: type of activation function to use
        :param inputs: whether this is input layer or not
        :return: list of perceptrons, representing a layer in the neural network
        """
        layer = list()
        for i in range(units):
            layer.append(Perceptron(activation, inputs))
        return layer

    def feedforward(self, inputs):
        """
        feeds forward the inputs to all of this layers perceptrons.
        :param inputs: the inputs to feed to the perceptrons of this layer
        :return: a vector containing the results of each perceptron's computations
        """
        results = list()
        for perceptron in self.units:
            results.append(perceptron.compute(inputs))
        return results


class Perceptron:
    def __init__(self, activation, inputs):
        """
        Creates a perceptron which utilizes the given activation function
        :param activation: type of activation function to use
        :param inputs: the inputs that this perceptron will be receiving
        """
        self.activation = activation
        self.inputs = inputs
        if inputs:
            self.weights = np.zeros(len(inputs.units))
        else:
            self.weights = None

    def compute(self, inputs):
        """
        Gets the sum of the products of each input along with its appropriate weight. Then feeds it through an
        activation function.
        :param inputs: the inputs to do computation on
        :return: the result, after having gone through an activation function
        """
        if self.weights is None:
            self.weights = np.arange(len(inputs))
        total = 0
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]
        # sigmoid
        if self.activation == "sigmoid":
            return total / (1 + abs(total))
        return -1


def one_hot(labels, depth):
    """
    Converts the passed in labels into one hot format
    :param labels: labels to be converted
    :param depth: number of possible classes
    :return: nparray of labels in one hot format
    """
    return None


