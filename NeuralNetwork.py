import numpy as np


class NeuralNetwork:
    def __init__(self, structure, activation, zeroes=True):
        """
        Creates skeleton of neural network
        :param structure: a list that gives the number of layers (length of list) and the number of units in those
        layers (numbers in list).
        :param activation: type of activation function to use throughout neural network
        :param zeroes: Whether to initialize all variables in the network to zero or not
        """
        self.hidden = list()
        for i, units in enumerate(structure):
            # input layer
            if i == 0:
                self.input = Layer("input", units, None, activation=activation)
            # output layer
            elif i == len(structure)-1:
                self.output = Layer("output", units, inputs=self.hidden[-1], activation=activation)
            # hidden layer
            else:
                # first hidden layer
                if not self.hidden:
                    self.hidden.append(Layer("hidden", units, inputs=self.input, activation=activation))
                # non-first hidden layer
                else:
                    self.hidden.append(Layer("hidden", units, inputs=self.hidden[i-2], activation=activation))


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


class Perceptron:
    def __init__(self, activation, inputs=None):
        """
        Creates a perceptron which utilizes the given activation function
        :param activation: type of activation function to use
        :param inputs: the inputs that this perceptron will be receiving
        """
        self.activation = activation
        self.inputs = inputs
        if inputs:
            self.weights = np.zeros(len(inputs.units))


x = NeuralNetwork([5, 10, 10, 2], "sigmoid")
print(x.hidden)
print(x.output.units[0].weights)
