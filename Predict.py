import argparse
import csv
from shutil import copyfile

import numpy as np
import os

import cv2
import time
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from FilesAndDirs import clear_directory
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
    if isinstance(filename, str):
        img = cv2.imread(filename)
    else:
        img = filename
    if img.shape[0] != IMAGE_HEIGHT or img.shape[1] != IMAGE_WIDTH:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def create_int_to_labels_map(name_to_int):
    int_to_name = {}
    for name in name_to_int:
        index = name_to_int[name]
        int_to_name[index] = name
    return int_to_name


def rotate(img):
    width = img.shape[1]
    height = img.shape[0]
    angle = np.random.rand() * 45
    scale = 1 + np.random.rand() * 1.1
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_AREA, borderMode=(cv2.BORDER_REFLECT_101))


def predict_with_tta(tta, robot_tta, images, model, tta_mode='mean'):
    tta_hflip = tta > 0
    tta_vflip = tta > 1
    tta_rotate = tta > 2

    results = []

    if robot_tta:
        tta_batch = []
        for image_file in images:
            data = load_image_data(image_file)
            tta_batch.append(data)
            augment(data, tta_batch, tta_hflip, tta_rotate, tta_vflip)

        probs_all = model.predict(np.array(tta_batch, dtype=np.float32))
        aggregate_ensemble_probs(probs_all, results, tta_mode)
    else:
        for image_file in images:
            tta_batch = []
            data = load_image_data(image_file)
            tta_batch.append(data)
            augment(data, tta_batch, tta_hflip, tta_rotate, tta_vflip)
            probs_all = model.predict(np.array(tta_batch, dtype=np.float32))

            # averaging probabilities
            aggregate_ensemble_probs(probs_all, results, tta_mode)

    return results


def aggregate_ensemble_probs(probs_all, results, tta_mode):
    if tta_mode == 'mean':
        results.append(np.mean(probs_all, axis=0))
    else:
        counts = np.sum(to_categorical(np.argmax(probs_all, axis=1), num_classes=len(probs_all[0])), axis=0)
        results.append(counts / np.sum(counts))


def augment(data, tta_batch, tta_hflip, tta_rotate, tta_vflip):
    if tta_hflip:
        tta_batch.append(cv2.flip(data, 0))
    if tta_vflip:
        tta_batch.append(cv2.flip(data, 1))
    if tta_rotate:
        tta_batch.append(rotate(data))


def sort_images(test_dir, tta, model, int_to_labels_map, tta_mode):
    """ Sort images with unknown labels to directories by predicted label"""

    # preparing sorted dir
    sorted_dir = os.path.join(test_dir, "sorted")
    if not os.path.exists(sorted_dir):
        os.mkdir(sorted_dir)
    else:
        clear_directory(sorted_dir)

    files = os.listdir(test_dir)
    image_files = []
    for file in files:
        path = os.path.join(test_dir, file)
        if not os.path.isdir(path):
            image_files.append(path)

    for i in range(len(image_files)):
        src_image = image_files[i]
        probs = predict_with_tta(tta, False, image_files[i:i+1], model, tta_mode)[0]

        src_image_name = os.path.split(src_image)[1]
        predicted_label_index = np.argmax(probs)
        prob = probs[predicted_label_index]
        if prob < 0.1:
            predicted_label = "unknown"
        else:
            predicted_label = int_to_labels_map[predicted_label_index]

        print("%s: %s" % (src_image_name, predicted_label))
        label_dir = os.path.join(sorted_dir, predicted_label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        dst_image_path = os.path.join(label_dir, src_image_name)
        copyfile(src_image, dst_image_path)

    return len(image_files)


def measure_accuracy(test_dir, tta, robot_tta, model, labels_map, int_to_labels_map, tta_mode):

    dirs = os.listdir(test_dir)
    test_dirs = []
    for d in dirs:
        path = os.path.join(test_dir, d)
        if not os.path.isdir(path) or d not in labels_map:
            # skipping files and unknown labels
            continue
        test_dirs.append(d)

    print("%d label directories found" % len(test_dirs))

    top_n = 5
    top_1_acc = 0
    top_5_acc = 0
    N = 0
    image_count = 0

    # arrays of prediction results for confusion maxtrix
    y_pred = []
    y_true = []

    for label_dir in test_dirs:

        print("\n%s/" % label_dir)

        # reading label dir
        label_path = os.path.join(test_dir, label_dir)
        files = os.listdir(label_path)
        image_files = []
        for f in files:
            path = os.path.join(label_path, f)
            if os.path.isfile(path):
                image_files.append(path)

        image_count += len(image_files)

        if robot_tta > 1:
            n_batches = len(image_files) // robot_tta
            for i in range(n_batches):
                offset = i * robot_tta
                robot_images = image_files[offset: offset + robot_tta]
                probs = predict_with_tta(tta, True, robot_images, model, tta_mode)[0]

                for im in robot_images:
                    print("\t%s" % os.path.split(im)[1])

                N, top_1_acc, top_5_acc = update_accuracy_counts(N, int_to_labels_map, label_dir, probs, top_1_acc,
                                                                 top_5_acc, top_n, y_pred, y_true)
        else:
            for img_file in image_files:
                print("\t%s" % os.path.split(img_file)[1])
                probs = predict_with_tta(tta, False, [img_file], model, tta_mode)[0]

                N, top_1_acc, top_5_acc = update_accuracy_counts(N, int_to_labels_map, label_dir, probs, top_1_acc,
                                                                 top_5_acc, top_n, y_pred, y_true)

    print("top 1 accuracy = %1.2f%%" % (top_1_acc*100.0/N))
    print("top 5 accuracy = %1.2f%%" % (top_5_acc*100.0/N))
    conf_matrix = confusion_matrix(y_true, y_pred)
    return image_count, conf_matrix


def update_accuracy_counts(N, int_to_labels_map, label_dir, probs, top_1_acc, top_5_acc, top_n, y_pred, y_true):
    sorted_indexes = np.argsort(probs)
    true_label = label_dir
    top_5_hit = False
    for t in range(top_n):
        label_index = sorted_indexes[-t - 1]
        label_prob = probs[label_index]
        label_name = int_to_labels_map[label_index]
        if t == 0 and label_name == true_label:
            top_1_acc += 1
            y_pred.append(label_name)
            y_true.append(true_label)
        if t < 5 and label_name == true_label:
            top_5_hit = True
        print("%1.2f%% %s" % (label_prob * 100.0, label_name))
    if top_5_hit:
        top_5_acc += 1
    N += 1
    return N, top_1_acc, top_5_acc


def load_model(data_root, name, snapshot):
    snapshot_path = os.path.join(data_root, SNAPSHOTS_PATH, name)
    labels_map = load_labels_to_int_mapping(os.path.join(snapshot_path, "labels.csv"))
    int_to_labels_map = create_int_to_labels_map(labels_map)
    model_obj = create_model_class(name)
    num_classes = len(labels_map)
    model = model_obj.create_model(IMAGE_WIDTH, IMAGE_HEIGHT, num_classes)
    model.load_weights(os.path.join(snapshot_path, snapshot), by_name=True)

    return model, labels_map, int_to_labels_map


def test(args):
    test_dir = os.path.join(args.data_root, args.image_dir)

    snapshot_path = os.path.join(args.data_root, SNAPSHOTS_PATH, args.model)
    labels_map = load_labels_to_int_mapping(os.path.join(snapshot_path, "labels.csv"))
    int_to_labels_map = create_int_to_labels_map(labels_map)

    num_classes = len(labels_map)
    print("%d classes for model %s" % (num_classes, args.model))

    # loading model
    model_obj = create_model_class(args.model)
    model = model_obj.create_model(IMAGE_WIDTH, IMAGE_HEIGHT, num_classes)
    print("loading weights...")
    model.load_weights(os.path.join(snapshot_path, args.snapshot), by_name=True)

    t = time.time()
    if args.mode.lower() == "sort":
        n = sort_images(test_dir, args.tta, model, int_to_labels_map, args.tta_mode)
    elif args.mode.lower() == "measure":
        n, cnf_matrix = measure_accuracy(test_dir, args.tta, args.rtta, model, labels_map, int_to_labels_map, args.tta_mode)
    else:
        print("Error: unknown mode: '" + args.mode + "'")
        return
    seconds_per_image = (time.time() - t)/n
    print("Processing took %1.1fms per image" % (seconds_per_image*1000.))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("image_dir", type=str, help="directory with images for prediction (relative path to data_root)")
    parser.add_argument("mode", type=str, default="measure", help="predict mode: measure or sort")
    parser.add_argument("--tta", type=int, default=0, help="0 - no TTA, 1 - hflip, 2 - vflip+hflip")
    parser.add_argument("--rtta", type=int, default=0, help="Robot TTA. <1 - no TTA, >1 - number of images to take for TTA")
    parser.add_argument("--tta_mode", type=str, default="mean", help="'mean' or 'majority' voting TTA")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--snapshot", type=str, help="snapshot")

    _args = parser.parse_args()
    test(_args)