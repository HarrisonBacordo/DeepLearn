import cv2
import numpy as np


# TODO find out how to do a sliding window function with numpy
class ConvLayer:

    def __init__(self, numfilters, window_size):
        self.numfilters = numfilters
        self.window_size = window_size
        self.filters = list()

        for i in range(numfilters):
            self.filters.append(np.random.uniform(-1, 1, window_size))

    def filter(self, _input):
        img = cv2.imread(_input)
        np.strid

