import cv2


def resize_to_resolution(im, downsample_size):
    if max(im.shape[0], im.shape[1]) > downsample_size:
        # downsampling
        if im.shape[0] > im.shape[1]:
            dsize = ((downsample_size * im.shape[1]) / im.shape[0], downsample_size)
        else:
            dsize = (downsample_size, (downsample_size * im.shape[0]) / im.shape[1])
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)

    return im

