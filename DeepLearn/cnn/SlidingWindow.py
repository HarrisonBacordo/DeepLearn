import DeepLearn.cnn as cnn
import numpy as np
import cv2


def sliding_window(_input, func,  _filter=None, window=None):
    """
    filters the given image and returns the resulting array
    :param _input: file path to image
    :param func: function to do on sliding window
    :param _filter: filter to slide across the image
    :param window: window to slide across image
    :return: the filtered array
    """
    if _filter is not None:
        arr = imgprocess(_input)
        box = _filter
    else:
        arr = _input
        box = window
    b_width = len(box[0])
    b_height = len(box)
    w_pad = np.zeros(np.size(arr, 0))[:, None]
    x = (np.size(arr, 0) % b_width) + 1
    for i in range(x):
        arr = np.hstack((arr, w_pad))

    h_pad = np.zeros(np.size(arr, 1))
    x = (np.size(arr, 1) % b_height) + 1
    for i in range(x):
        arr = np.vstack((arr, h_pad))

    cmprsd = np.zeros((int(np.size(arr, 1) / np.size(box, 1)), int(np.size(arr, 1) / np.size(box, 1))))
    for i in range(int(np.size(arr, 0) / b_width)):
        for j in range(int(np.size(arr, 1) / b_height)):
            w_start = i * b_width
            w_end = w_start + b_width
            h_start = j * b_height
            h_end = h_start + b_height
            window = func(arr[w_start:w_end, h_start:h_end], box)
            if _filter is None:
                cmprsd[i, j] = window
            else:
                arr[w_start:w_end, h_start:h_end] = window
    if _filter is None:
        return cmprsd
    else:
        return arr

def imgprocess(img):
    if type(img) is not str:
        return img
    im = cv2.imread(img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im