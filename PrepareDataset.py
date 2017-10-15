import argparse

import os
from multiprocessing import Pool

import numpy as np
import cv2

from FilesAndDirs import get_n_pass_image_file_name, get_mask_file_name_for_image, get_image_names_from_dir, \
    get_downsampled_dir, get_downsampled_img_name, get_masks_dir, get_default_mask_file_name, get_raw_dir
from ImageUtils import resize_to_resolution

IMG_RESOLUTION = 1024


def downsample_image(img_file, out_name):
    img = cv2.imread(img_file)
    img_small = resize_to_resolution(img, IMG_RESOLUTION)
    cv2.imwrite(out_name, img_small)
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


def create_default_mask(img_file, mask_dst_file):
    rgb = cv2.imread(get_downsampled_img_name(img_file))
    hls = cv2.cvtColor(rgb, cv2.COLOR_BGR2HLS)
    h = cv2.split(hls)[0]
    k = 2.0 * 3.14159 / 180
    float_h = h.astype(np.float32) * k
    cos_h = np.cos(float_h)
    sin_h = np.sin(float_h)
    cos_sin_h = cv2.merge([cos_h, sin_h])

    color_key = np.array([[163, 73, 164]], dtype=np.uint8)
    color_key_hls = cv2.cvtColor(color_key, cv2.COLOR_BGR2HLS)
    hue_key = cv2.split(color_key_hls)[0].astype(np.float32) * k
    key_cos_h = np.cos(hue_key)
    key_sin_h = np.sin(hue_key)
    key_cos_sin = cv2.merqe([key_cos_h, key_sin_h])

    diff = cos_sin_h - key_cos_sin

    pass


def downsample_images(pool, data_dir, image_files):

    downsampled_dir = get_downsampled_dir(data_dir)
    if not os.path.exists(downsampled_dir):
        print "creating dir: %s" % downsampled_dir
        os.mkdir(downsampled_dir)

    futures = []

    for img_file in image_files:
        downsampled_file = get_downsampled_img_name(img_file)
        if not os.path.exists(downsampled_file):
            futures.append(pool.apply_async(downsample_image, (img_file, downsampled_file)))

    for future in futures:
        name = future.get()
        print "downsampled: %s" % name


def generate_default_masks(pool, data_dir, image_files):

    masks_dir = get_masks_dir(data_dir)
    if not os.path.exists(masks_dir):
        print "creating dir: %s" % masks_dir
        os.mkdir(masks_dir)

    futures = []

    for img_file in image_files:
        mask_file = get_default_mask_file_name(img_file)
        if not os.path.exists(mask_file):
            futures.append(pool.apply_async(create_default_mask, (img_file, mask_file)))

    for future in futures:
        name = future.get()
        print "default mask: %s" % name


def segment_images(pool, data_dir, image_files):
    pass


def prepare(args):

    image_files = get_image_names_from_dir(get_raw_dir(args.data_dir))

    print "%d images found" % len(image_files)

    pool = Pool()

    print "downsampling..."
    downsample_images(pool, args.data_dir, image_files)
    print "downsampling done"

    print "generating default masks..."
    generate_default_masks(pool, args.data_dir, image_files)
    print "done"

    print "segmenting images..."
    segment_images(pool, args.data_dir, image_files)
    print "segmenting done"

    out_dir = os.path.join(args.data_dir, "out")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("data_dir", type=str, help="raw image dir")

    _args = parser.parse_args()
    prepare(_args)