import argparse

import os
from multiprocessing import Pool

import numpy as np
import cv2

from FilesAndDirs import get_n_pass_image_file_name, get_image_names_from_dir, \
    get_downsampled_dir, get_downsampled_img_name, get_masks_dir, get_mask_file_name, get_raw_dir, get_segmentation_dir, \
    create_dir, get_seg_file_name, get_parts_dir, get_parts_dir_name, clear_directory
from ImageUtils import resize_to_resolution, shrink_mask, expand_mask

IMG_RESOLUTION = 1024


def downsample_image(img_file, out_name):
    img = cv2.imread(img_file)
    img_small = resize_to_resolution(img, IMG_RESOLUTION)
    cv2.imwrite(out_name, img_small)
    return img_file


def create_segmentation(img_file, seg_file):
    try:
        img = cv2.imread(get_downsampled_img_name(img_file))

        # loading mask
        mask_img = cv2.imread(get_mask_file_name(img_file))

        # converting to grayscale
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        mask = np.ones(img.shape[:2], np.uint8) * cv2.GC_PR_BGD
        mask[mask_img == 0] = cv2.GC_BGD
        mask[mask_img == 255] = cv2.GC_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented = img * mask2[:, :, np.newaxis]

        # saving segmentation
        cv2.imwrite(seg_file, segmented)
        return img_file, True

    except Exception as _:
        return img_file, False


def split_parts_for_image(img_file, out_dir):
    try:

        if os.path.exists(out_dir):
            clear_directory(out_dir)
        else:
            create_dir(out_dir)

        segmented_img = cv2.imread(get_seg_file_name(img_file))
        width = segmented_img.shape[1]
        height = segmented_img.shape[0]

        min_area = 10
        max_area = width * height / 2

        mask = cv2.cvtColor((segmented_img != 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        part_index = 0

        while True:
            nz = np.nonzero(mask.flatten())[0].flatten()
            if len(nz) == 0:
                break

            nz_i = 0
            found_mask = None
            found_image = None
            while True:
                index = nz[nz_i]
                seed_x = index % width
                seed_y = index / width

                ff_mask = np.zeros((height+2, width+2), dtype=np.uint8)
                area, rect = cv2.floodFill(mask, ff_mask, (seed_x, seed_y), 255, flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]

                # slicing into found rect
                roi = mask[y:y+h, x:x+w]
                roi_mask = ff_mask[y+1:y+1+h, x+1:x+1+w]

                found = False
                if min_area < area < max_area:
                    found_mask = ff_mask
                    found_image = segmented_img[y:y+h, x:x+w][:]
                    found_image[roi_mask == 0] = 0
                    found = True

                # clearing found component in the mask
                roi[roi_mask != 0] = 0

                if found:
                    break

                nz_i += 1
                if nz_i >= len(nz):
                    break

            if found_mask is not None:
                # we found some part
                part_file = os.path.join(out_dir, "part_%02d.png" % part_index)
                cv2.imwrite(part_file, found_image)
                part_index += 1
                # print "#%d: area = %d, rect = %s" % (part_index, area, str(rect))

        return img_file, True
    except Exception as _:
        return img_file, False


def calc_color_key_features(rgb):
    hls = cv2.cvtColor(rgb.astype(np.float32)/255.0, cv2.COLOR_BGR2HLS)
    hls_split = cv2.split(hls)
    h = hls_split[0]
    l = hls_split[1]
    s = hls_split[2]
    k = 3.14159 / 180
    float_h = h.astype(np.float32) * k
    cos_h = np.cos(float_h)
    sin_h = np.sin(float_h)
    features_img = cv2.merge([cos_h, sin_h, (l * 0.2/255).astype(np.float32), (s * 0.8/255).astype(np.float32)])
    return features_img, hls_split


def calc_color_key_features_lab(rgb):
    return calc_color_key_features(rgb)
    # lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    # lab_split = cv2.split(lab)
    # l = lab_split[0]
    # a = lab_split[1]
    # b = lab_split[2]
    # features_img = cv2.merge(np.array([l*0.01, a, b], dtype=np.float32))
    # return features_img, lab_split


def create_default_mask(img_file, mask_dst_file):
    rgb = cv2.imread(get_downsampled_img_name(img_file))

    img_features, img_hls = calc_color_key_features(rgb)

    color_key1 = np.array([[163, 73, 164]], dtype=np.uint8).reshape((1,1,3))
    key_features1, _ = calc_color_key_features(color_key1)

    color_key2 = np.array([[200, 170, 200]], dtype=np.uint8).reshape((1,1,3))
    key_features2, _ = calc_color_key_features(color_key2)

    color_key3 = np.array([[140, 80, 140]], dtype=np.uint8).reshape((1,1,3))
    key_features3, _ = calc_color_key_features(color_key3)

    diff1 = np.linalg.norm(img_features - key_features1, axis=2)
    diff2 = np.linalg.norm(img_features - key_features2, axis=2)
    diff3 = np.linalg.norm(img_features - key_features3, axis=2)
    diff = np.minimum(np.minimum(diff1, diff2), diff3)

    diff_file = os.path.splitext(mask_dst_file)[0] + '_diff.png'
    k = 250.0 / np.max(diff)
    diff8 = (diff * k).astype(np.uint8)
    cv2.imwrite(diff_file, diff8)

    MAX_BG_DIFF = 0.2
    MIN_FG_DIFF = 0.8

    # using hue distances only for valid pixels:
    # no color or lightness saturations
    MIN_LIGHTNESS = 0.3
    MIN_COLOR_SATURATION = 0.1
    l = img_hls[1]  # lightness
    s = img_hls[2]  # saturation
    valid_pixels = (l > MIN_LIGHTNESS) * (s > MIN_COLOR_SATURATION)

    valid_file = os.path.splitext(mask_dst_file)[0] + '_valid.png'
    cv2.imwrite(valid_file, valid_pixels.astype(np.uint8) * 255)

    bg_mask = (diff < MAX_BG_DIFF) * valid_pixels
    fg_mask = (diff > MIN_FG_DIFF) * valid_pixels
    # unknown_mask = ~(bg_mask + fg_mask)
    # unknown_mask = expand_mask(unknown_mask.astype(np.uint8), 1)
    # bg_mask = bg_mask * (~unknown_mask)
    # fb_mask = fg_mask * (~unknown_mask)

    rgb[bg_mask] = np.array([0,0,0], dtype=np.uint8)
    rgb[fg_mask] = np.array([255,255,255], dtype=np.uint8)

    cv2.imwrite(mask_dst_file, rgb)
    return img_file


def downsample_images(pool, data_dir, image_files):

    downsampled_dir = get_downsampled_dir(data_dir)
    create_dir(downsampled_dir)

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
    create_dir(masks_dir)

    futures = []

    for img_file in image_files:
        mask_file = get_mask_file_name(img_file)
        if not os.path.exists(mask_file):
            futures.append(pool.apply_async(create_default_mask, (img_file, mask_file)))

    for future in futures:
        name = future.get()
        print "default mask: %s" % name


def segment_images(pool, data_dir, image_files):

    seg_dir = get_segmentation_dir(data_dir)
    create_dir(seg_dir)

    futures = []

    for img_file in image_files:
        seg_file = get_seg_file_name(img_file)
        if not os.path.exists(seg_file):
            futures.append(pool.apply_async(create_segmentation, (img_file, seg_file)))

    for future in futures:
        name, success = future.get()
        print "segmented: %s" % name if success else "failed to segment: %s" % name


def split_parts(pool, data_dir, image_files):

    parts_dir = get_parts_dir(data_dir)
    create_dir(parts_dir)

    futures = []

    for img_file in image_files:
        parts_dir = get_parts_dir_name(img_file)
        if not os.path.exists(parts_dir):
            futures.append(pool.apply_async(split_parts_for_image, (img_file, parts_dir)))

    for future in futures:
        name, success = future.get()
        print "parts split: %s" % name if success else "failed to split parts: %s" % name


def prepare(args):

    image_files = get_image_names_from_dir(get_raw_dir(args.data_dir))

    print "%d images found" % len(image_files)

    pool = Pool(1)

    print "downsampling..."
    downsample_images(pool, args.data_dir, image_files)
    print "downsampling done"

    print "generating default masks..."
    generate_default_masks(pool, args.data_dir, image_files)
    print "done"

    print "segmenting images..."
    segment_images(pool, args.data_dir, image_files)
    print "segmenting done"

    print "splitting parts..."
    split_parts(pool, args.data_dir, image_files)
    print "splitting parts done"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("data_dir", type=str, help="raw image dir")

    _args = parser.parse_args()
    prepare(_args)