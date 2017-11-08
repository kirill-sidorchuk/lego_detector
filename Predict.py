import argparse
import csv
import numpy as np
import os

import cv2
from keras.engine import Model

from Finetune import IMAGE_HEIGHT, IMAGE_WIDTH
from ModelUtils import create_model_class

TEST_DIR = "test"
SNAPSHOTS_PATH = "snapshots"
TRUE_LABELS = "labels.txt"


def load_labels_to_int_mapping(filename):
    label_map = {}
    with open(filename, "r") as f:
        content = csv.reader(f)
        for row in content:
            label_map[row[0]] = int(row[1])

    return label_map


def parse_images_labels(files):
    file_labels = {}
    for filename in files:
        name = os.path.splitext(filename)[0]
        i = name.find('-')
        if i != -1:
            name = name[0:i]
        file_labels[filename] = name
    return file_labels


def load_image_data(filename):
    img = cv2.imread(filename)
    if img.shape[0] != IMAGE_HEIGHT or img.shape[1] != IMAGE_WIDTH:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def create_int_to_labels_map(name_to_int):
    int_to_name = {}
    for name in name_to_int:
        index = name_to_int[name]
        int_to_name[index] = name
    return int_to_name


def test(args):
    test_dir = os.path.join(args.data_root, TEST_DIR)

    snapshot_path = os.path.join(args.data_root, SNAPSHOTS_PATH, args.model)
    labels_map = load_labels_to_int_mapping(os.path.join(snapshot_path, "labels.csv"))
    int_to_labels_map = create_int_to_labels_map(labels_map)

    num_classes = len(labels_map)
    print("%d classes for model %s" % (num_classes, args.model))

    # reading test directory
    test_images = os.listdir(test_dir)
    print("%d test images found" % len(test_images))
    test_images_labels = parse_images_labels(test_images)

    # loading model
    model_obj = create_model_class(args.model)
    model = model_obj.create_model(IMAGE_WIDTH, IMAGE_HEIGHT, num_classes)
    print("loading weights...")
    model.load_weights(os.path.join(snapshot_path, args.snapshot), by_name=True)

    for filename in test_images:
        data = load_image_data(os.path.join(test_dir, filename))
        probs = model.predict(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))
        sample_index = 0
        sample_probs = probs[sample_index]
        label_index = np.argmax(sample_probs)
        label_prob = sample_probs[label_index]
        label_name = int_to_labels_map[label_index]
        print("%s: %s (%1.2f%%)" % (filename, label_name, label_prob*100.0))

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--snapshot", type=str, help="snapshot")

    _args = parser.parse_args()
    test(_args)