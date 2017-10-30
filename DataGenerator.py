import os
import numpy as np
import cv2

from FinalizeDataset import SORTED_DIR


class DataGenerator(object):

    def __init__(self, data_root, dataset_type):
        self.data_root = data_root
        self.image_files = self.load_labels(dataset_type + ".txt")
        self.images = self.load_images()

    def load_labels(self, labels_file):
        with open(os.path.join(self.data_root, labels_file), "r") as f:
            lines = f.readlines()

        images = {}
        for line in lines:
            path = line.strip()
            if len(path) < 1:
                continue
            label = os.path.split(path)[0]
            img_file = os.path.join(self.data_root, SORTED_DIR, path)
            if label in images:
                images[label].append(img_file)
            else:
                images[label] = [img_file]

        return images

    def load_images(self):
        images = {}
        for _class in self.image_files:
            imgs = self.image_files[_class]
            for img_file in imgs:
                # load image
                img = cv2.imread(img_file)

                # create mask
                img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_mask = (img_bw == 0).astype(np.uint8)

                if _class in self.images:
                    self.images[_class].append((img, img_mask))
                else:
                    self.images[_class] = [(img, img_mask)]

        return images

