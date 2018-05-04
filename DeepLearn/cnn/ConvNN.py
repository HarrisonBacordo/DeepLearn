import numpy as np

from DeepLearn.cnn import ConvLayer


def relu(arr):
    """
    implements relu function on given array
    :param arr: array to apply relu to
    :return: array with relu applied
    """
    return np.maximum(arr, 0)



