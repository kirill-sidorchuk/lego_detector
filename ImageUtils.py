import cv2
import numpy as np

def resize_to_resolution(im, downsample_size):
    if max(im.shape[0], im.shape[1]) > downsample_size:
        # downsampling
        if im.shape[0] > im.shape[1]:
            dsize = ((downsample_size * im.shape[1]) // im.shape[0], downsample_size)
        else:
            dsize = (downsample_size, (downsample_size * im.shape[0]) // im.shape[1])
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)

    return im


def expand_mask(mask, kernel_size, n_iter=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=n_iter)


def shrink_mask(mask, kernel_size, n_iter=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=n_iter)
