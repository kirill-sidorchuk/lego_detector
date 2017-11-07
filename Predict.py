import argparse
import csv

import os

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


def test(args):
    test_dir = os.path.join(args.data_root, TEST_DIR)

    snapshot_path = os.path.join(args.data_root, SNAPSHOTS_PATH, args.model)
    labels_map = load_labels_to_int_mapping(os.path.join(snapshot_path, "labels.csv"))

    print("%d classes for model %s" % (len(labels_map), args.model))

    # reading test directory
    test_images = os.listdir(test_dir)
    print("%d test images found" % len(test_images))

    # loading model



    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--snapshot", type=str, help="snapshot")

    _args = parser.parse_args()
    test(_args)