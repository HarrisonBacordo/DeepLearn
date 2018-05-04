from DeepLearn.cnn import SlidingWindow as sw
import numpy as np

class PoolLayer:

    def __init__(self, inputs, window_size, stride):
        self.inputs = inputs
        self.window = np.zeros(window_size)
        self.stride = stride

    def execute(self):
        container = list()
        for input_ in self.inputs:

            cmprsd = sw.sliding_window(input_, pool, window=self.window)
            container.append(cmprsd)
        return container


def pool(img, window):
    """
    Slides the filter over the given indices of the image and returns the resulting 2d array
    :param img: portion of the image to filter
    :param window: the window to apply to the given image
    :return: the resulting filtered portion of the image
    """
    # ENSURE IMG AND FILTER ARE SAME SHAPE
    assert img.shape == window.shape
    # ELEMENT-WISE MULTIPLICATION ON ALL ELEMENTS
    return np.amax(img)
