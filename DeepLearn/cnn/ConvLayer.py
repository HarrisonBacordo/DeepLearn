from DeepLearn.cnn import SlidingWindow, ConvNN
import cv2
import numpy as np


# TODO find out how to do a sliding window function with numpy
class ConvLayer:

    def __init__(self, inputs, numfilters, window_size):
        self.inputs = inputs
        self.numfilters = numfilters
        self.window_size = window_size
        self.filters = list()

        for i in range(numfilters):
            self.filters.append(np.random.uniform(-1, 1, window_size))

    def execute(self):
        container = list()
        for input_ in self.inputs:
            for _filter in self.filters:
                filt = SlidingWindow.sliding_window(input_, filter_, _filter=_filter)
                filt = ConvNN.relu(filt)
                container.append(filt)
        return container


def filter_(img, _filter):
    """
    Slides the filter over the given indices of the image and returns the resulting 2d array
    :param img: portion of the image to filter
    :param _filter: the filter to apply to the given image
    :return: the resulting filtered portion of the image
    """
    # ENSURE IMG AND FILTER ARE SAME SHAPE
    assert img.shape == _filter.shape
    # ELEMENT-WISE MULTIPLICATION ON ALL ELEMENTS
    return np.multiply(img, _filter)