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
        file_labels[filename] = parse_filename_label(name)
    return file_labels


def parse_filename_label(name):
    i = name.find('-')
    if i != -1:
        name = name[0:i]
    return name


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

    tta_hflip = True
    tta_vflip = True

    top_n = 5
    top_1_acc = 0
    top_5_acc = 0
    for filename in test_images:
        tta_batch = []
        data = load_image_data(os.path.join(test_dir, filename))
        tta_batch.append(data)
        if tta_hflip:
            tta_batch.append(cv2.flip(data, 0))
        if tta_vflip:
            tta_batch.append(cv2.flip(data, 1))
        probs_all = model.predict(np.array(tta_batch, dtype=np.float32))

        # averaging probabilities
        probs = np.mean(probs_all, axis=0)

        sorted_indexes = np.argsort(probs)
        print("\n%s:" % filename)
        true_label = parse_filename_label(os.path.splitext(filename)[0])
        top_5_hit = False
        for t in range(top_n):
            label_index = sorted_indexes[-t-1]
            label_prob = probs[label_index]
            label_name = int_to_labels_map[label_index]
            if t == 0 and label_name == true_label:
                top_1_acc += 1
            if t < 5 and label_name == true_label:
                top_5_hit = True
            print("%1.2f%% %s" % (label_prob*100.0, label_name))

        if top_5_hit:
            top_5_acc += 1

    print("top 1 accuracy = %1.2f%%" % (top_1_acc*100.0/len(test_images)))
    print("top 5 accuracy = %1.2f%%" % (top_5_acc*100.0/len(test_images)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--snapshot", type=str, help="snapshot")

    _args = parser.parse_args()
    test(_args)