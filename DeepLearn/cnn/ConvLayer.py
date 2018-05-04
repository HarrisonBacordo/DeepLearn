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

    def sliding_window(self, _input, _filter):
        """
        filters the given image and returns the resulting array
        :param _input: file path to image
        :param filter: filter to slide across the image
        :return: the filtered img array
        """
        f_width = len(_filter[0])
        f_height = len(_filter)
        img = cv2.imread("test.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w_pad = np.zeros(np.size(img, 0))[:, None]
        x = (np.size(img, 0) % f_width) + 1
        for i in range(x):
            img = np.hstack((img, w_pad))

        h_pad = np.zeros(np.size(img, 1))
        x = (np.size(img, 1) % f_height) + 1
        for i in range(x):
            img = np.vstack((img, h_pad))

        for i in range(int(np.size(img, 0) / f_width)):
            for j in range(int(np.size(img, 1) / f_height)):
                w_start = i * f_width
                w_end = w_start + f_width
                h_start = j * f_height
                h_end = h_start + f_height
                window = self.filter(img[w_start:w_end, h_start:h_end], _filter)
                img[w_start:w_end, h_start:h_end] = window
        return img

    def filter(self, img, _filter):
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

    def execute(self):
        container = list()
        for _filter in self.filters:
            filt = self.sliding_window("test.png", _filter)
            filt = relu(filt)
            container.append(filt)
        return container