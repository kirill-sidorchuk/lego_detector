import argparse

import os
from multiprocessing import Pool

import numpy as np
import cv2

IMG_RESOLUTION = 1024


def create_mask(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    pass


def prepare_image_pass1(img_file, out_dir):
    img = cv2.imread(img_file)
    img_small = resize_to_resolution(img, IMG_RESOLUTION)
    out_file = os.path.split(img_file)[1]
    out_file = os.path.splitext(out_file)[0] + '.png'
    out_file = os.path.join(out_dir, out_file)

    cv2.imwrite(out_file, img_small)
    return img_file


def prepare_image_pass2(img_file, out_dir):
    try:
        img = cv2.imread(img_file)
        img_small = resize_to_resolution(img, IMG_RESOLUTION)

        # composing mask name
        mask_file_path = get_mask_file_name_for_image(img_file)
        mask_img = cv2.imread(mask_file_path)
        # converting to grayscale
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        mask = np.ones(img_small.shape[:2], np.uint8) * cv2.GC_PR_BGD
        mask[mask_img == 0] = cv2.GC_BGD
        mask[mask_img == 255] = cv2.GC_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented = img_small * mask2[:, :, np.newaxis]

        out_file = os.path.split(img_file)[1]
        out_file = os.path.splitext(out_file)[0] + '_pass2.png'
        out_file = os.path.join(out_dir, out_file)
        cv2.imwrite(out_file, segmented)
        return img_file, True

    except Exception as _:
        return img_file, False


def prepare_image_pass3(img_file, out_dir):
    try:
        pass_2_image_name = get_n_pass_image_file_name(img_file, out_dir, 2)

        segmented_img = cv2.imread(pass_2_image_name)
        width = segmented_img.shape[1]
        height = segmented_img.shape[0]

        min_area = 10
        max_area = width * height / 2

        mask = (segmented_img != 0).astype(np.uint8)

        while True:
            nz = np.nonzero(mask.flatten())[0].flatten()
            if len(nz) == 0:
                break

            nz_i = 0
            found_mask = None
            found_rect = None
            while True:
                index = nz[nz_i]
                seed_x = index % width
                seed_y = index / width

                ff_mask = np.zeros((height+2, width+2), dtype=np.uint8)
                area, rect = cv2.floodFill(mask, ff_mask, (seed_x, seed_y), 255, flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

                if min_area < area < max_area:
                    found_mask = ff_mask
                    found_rect = rect
                    break

                nz_i += 1

            print "area = %d, rect = %s" % (area, str(rect))

            pass


        return img_file, True
    except Exception as _:
        return img_file, False


def get_mask_file_name_for_image(img_file):
    img_dir, img_file = os.path.split(img_file)
    mask_dir = os.path.join(img_dir, "masks")
    mask_file_name = os.path.splitext(img_file)[0] + '.png'
    mask_file_path = os.path.join(mask_dir, mask_file_name)
    return mask_file_path


def get_n_pass_image_file_name(img_file, out_dir, n):
    img_file = os.path.split(img_file)[1]
    img_file = os.path.splitext(img_file)[0]
    pass_n_name = os.path.join(out_dir, "%s_pass%d.png" % (img_file, n))
    return pass_n_name


def prepare(args):
    image_files = get_image_names_from_dir(args.image_dir)

    print "%d images found" % len(image_files)

    out_dir = os.path.join(args.image_dir, "out")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pool = Pool()
    futures = []

    # # starting pass1
    # for img_file in image_files:
    #     if not os.path.exists(get_mask_file_name_for_image(img_file)):
    #         futures.append(pool.apply_async(prepare_image_pass1, (img_file, out_dir)))
    #
    # for future in futures:
    #     # try:
    #     name = future.get()
    #     print "pass1: %s" % name
    #
    # # starting pass2
    # futures = []
    #
    # for img_file in image_files:
    #     if os.path.exists(get_mask_file_name_for_image(img_file)):
    #         futures.append(pool.apply_async(prepare_image_pass2, (img_file, out_dir)))
    #
    # for future in futures:
    #     name, success = future.get()
    #     print "pass2: %s - %s" % (name, "ok" if success else "fail")

    # starting pass3
    futures = []

    for img_file in image_files:
        if os.path.exists(get_n_pass_image_file_name(img_file, out_dir, 2)):
            futures.append(pool.apply_async(prepare_image_pass3, (img_file, out_dir)))

    for future in futures:
        name, success = future.get()
        print "pass3: %s - %s" % (name, "ok" if success else "fail")


def get_image_names_from_dir(image_dir):
    _files = os.listdir(image_dir)
    image_files = []
    for f in _files:
        if f.lower().endswith('.jpg'):
            image_files.append(os.path.join(image_dir, f))
    return image_files


def resize_to_resolution(im, downsample_size):
    if max(im.shape[0], im.shape[1]) > downsample_size:
        # downsampling
        if im.shape[0] > im.shape[1]:
            dsize = ((downsample_size * im.shape[1]) / im.shape[0], downsample_size)
        else:
            dsize = (downsample_size, (downsample_size * im.shape[0]) / im.shape[1])
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)

    return im


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("image_dir", type=str, help="raw image dir")

    _args = parser.parse_args()
    prepare(_args)