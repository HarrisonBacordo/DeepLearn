import numpy as np


def sigmoid(x, deriv=False):
    """
    implements sigmoid on num
    :param x: number to implement sigmoid on
    :param deriv: calculate derivative
    :return: result from sigmoid calculation
    """
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def bipolar_sigmoid(x, deriv=False):
    """
    implements bipolar sigmoid on num
    :param x: number to implement bipolar sigmoid on
    :param deriv: calculate derivative
    :return: result from bipolar sigmoid calculation
    """
    if deriv:
        return x * (1 - x)
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def binarystep(x, deriv=False):
    """
    implements binary step on num
    :param x: number to implement binary step on
    :param deriv: calculate derivative
    :return:result from binary step calculation
    """
    if deriv:
        return 1
    else:
        if x > 0:
            return 1
        return 0


def tanh(x, deriv=False):
    """
    implements tanh function on num
    :param x: number to implement tanh on
    :param deriv: calculate derivative
    :return: result from tanh calculation
    """
    if deriv:
        return 1
    return np.tanh(x)


def arctan(x, deriv=False):
    """
    implements arctan function on num
    :param x: number to implement arctan on
    :param deriv: calculate derivative
    :return: result from arctan calculation
    """
    if deriv:
        return 1
    return np.arctan(x)


def lecun_tanh(x, deriv=False):
    """
    implements Lecun's Tanh function on num
    :param x: number to implement Lecun's Tanh on
    :param deriv: calculate derivative
    :return: result from Lecun's Tanh calculation
    """
    if deriv:
        return 1
    return 1.7159 * np.tanh(2/3 * x)


def relu(x, deriv=False):
    """
    implements ReLU function on num
    :param x: number to implement ReLU on
    :param deriv: calculate derivative
    :return: result from ReLU calculation
    """
    if deriv:
        return np.greater(x, 0).astype(int)
    return np.maximum(0, x)


def leaky_relu(x, deriv=False):
    """
    implements Leaky ReLU function on num
    :param x: number to implement Leaky ReLU on
    :param deriv: calculate derivative
    :return: result from Leaky ReLU calculation
    """
    if deriv:
        return 1
    return np.maximum(0.01 * x, x)


def smooth_relu(x, deriv=False):
    """
    implements Smooth ReLU function on num
    :param x: number to implement Smooth ReLU on
    :param deriv: calculate derivative
    :return: result from Smooth ReLU calculation
    """
    if deriv:
        return 1
    return np.log(1 + np.exp(x))


def logit(x, deriv=False):
    """
    implements logit function on num
    :param x: number to implement logit function on
    :param deriv: calculate derivative
    :return: result from logit calculation
    """
    if deriv:
        return 1
    return np.log(x / (1 - x))


def softmax(x, deriv=False):
    """
        implements softmax function on num
        :param x: number to implement softmax function on
        :param deriv: calculate derivative
        :return: result from softmax calculation
        """
    if deriv:
        return 1
    return np.exp(x) / np.sum(np.exp(x))
