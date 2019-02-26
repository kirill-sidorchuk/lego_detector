import argparse

import os

SORTED_DIR = "sorted"
BACKGROUNDS_DIR = "backgrounds"

TRAIN_SPLIT = 0.7


def write_labels(data_root, dataset_type, data):
    label_file = os.path.join(data_root, dataset_type + ".txt")
    with open(label_file, "w") as f:
        for _class in data:
            imgs = data[_class]
            for img in imgs:
                img_file = os.path.join(_class, img)
                f.write("%s\n" % img_file.replace('\\', '/'))


def finalize(args):

    sorted_dir = os.path.join(args.data_root, SORTED_DIR)
    class_dir = os.listdir(sorted_dir)
    print("%d classes found" % len(class_dir))

    class_images = {}

    total_num_images = 0

    for _dir in class_dir:
        subdir = os.path.join(sorted_dir, _dir)
        img_files = os.listdir(subdir)
        print("%s: %d images" % (_dir, len(img_files)))
        class_images[_dir] = img_files
        total_num_images += len(img_files)

    print("total number of images: %d" % total_num_images)

    # creating train/validation split

    train_data = {}
    val_data = {}

    n_train_images = 0
    n_val_images = 0

    for _class in class_images:
        imgs = class_images[_class]
        n_train = int(len(imgs) * TRAIN_SPLIT)
        train_data[_class] = imgs[0: n_train]
        val_data[_class] = imgs[n_train:]
        n_train_images += n_train
        n_val_images += len(imgs) - n_train

    print("train data: %d images" % n_train_images)
    print("validation data: %d images" % n_val_images)

    # writing label files
    write_labels(args.data_root, "train", train_data)
    write_labels(args.data_root, "val", val_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument("data_root", type=str, help="data root directory")

    _args = parser.parse_args()
    finalize(_args)