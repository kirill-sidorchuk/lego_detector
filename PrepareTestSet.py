import argparse

import os
from multiprocessing.pool import Pool

import cv2

from Finetune import IMAGE_WIDTH
from ImageUtils import resize_to_resolution


def prepare_image(src_file, dst_file):
    img = cv2.imread(src_file)
    if img is None:
        return os.path.split(src_file)[1], False

    img = resize_to_resolution(img, IMAGE_WIDTH * 2)

    # cropping central part
    center = (img.shape[1]//2, img.shape[0]//2)
    crop_w = img.shape[1]//2
    crop_h = crop_w
    x0 = (img.shape[1] - crop_w)//2
    y0 = (img.shape[0] - crop_h)//2
    crop = img[y0:y0 + crop_h, x0:x0 + crop_w]

    cv2.imwrite(dst_file, crop)
    return os.path.split(src_file)[1], True


def prepare(args):

    image_files = os.listdir(args.raw_dir)
    print("%d images found" % len(image_files))

    out_dir = os.path.join(args.raw_dir, "prepared")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pool = Pool(4)
    futures = []

    for img_file in image_files:
        full_img_path = os.path.join(args.raw_dir, img_file)
        if os.path.isdir(full_img_path):
            continue
        futures.append(pool.apply_async(prepare_image, (full_img_path, os.path.join(out_dir, img_file))))

    # waiting for results
    for future in futures:
        try:
            name, success = future.get()
            print("%s : %s" % (name, "ok" if success else "failed"))
        except Exception as e:
            print("Exception in prepare_image: " + str(e))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("raw_dir", type=str, help="raw image dir")

    _args = parser.parse_args()
    prepare(_args)