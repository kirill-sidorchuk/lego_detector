import os
import numpy as np
import cv2
from keras.utils import to_categorical

import ImageUtils
from FinalizeDataset import SORTED_DIR

BG_RESOLUTION = 2048


class DataGenerator(object):

    def __init__(self, data_root, dataset_type, max_rotation, max_zoom, max_saturation_delta, max_lightness_delta,\
                 batch_size, image_size):
        self.data_root = data_root
        self.max_rotation = max_rotation
        self.max_zoom = max_zoom
        self.max_saturation_delta = max_saturation_delta
        self.max_lightness_delta = max_lightness_delta
        self.image_size = image_size
        self.batch_size = batch_size

        self.image_files = self.load_labels(dataset_type + ".txt")
        self.images = self.load_images()
        self.num_classes = len(self.images)
        self.image_tuples = self.initialize_image_order()

        self.bg_images = self.load_backgrounds()

    def load_backgrounds(self):
        bg_dir = os.path.join(self.data_root, "backgrounds")
        bg_files = os.listdir(bg_dir)
        bg_images = []
        for bg_file in bg_files:
            bg_img = cv2.imread(os.path.join(bg_dir, bg_file))
            if bg_img is None:
                continue
            bg_resized = ImageUtils.resize_to_resolution(bg_img, BG_RESOLUTION)
            bg_images.append(bg_resized)
        return bg_images

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

    def initialize_image_order(self):
        image_tuples = []
        for _class in self.images:
            imgs = self.images[_class]
            for img in imgs:
                image_tuples.append((img[0], img[1], _class))
        return image_tuples

    def render(self, fg_image, fg_mask, bg_image):
        dst = bg_image.copy()
        dst[fg_mask] = fg_image
        return dst

    def transform_image(self, img, dst_size, mask=None):

        # random rotation
        center = (img.shape[1]/2, img.shape[0]/2)
        angle = np.random.rand() * self.max_rotation
        scale = np.random.rand() * self.max_zoom
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.wrapAffine(img, mat, dst_size, flags=cv2.INTER_AREA, border_mode=cv2.BORDER_REFLECT_101)

        if mask is not None:
            rotated_mask = cv2.wrapAffine(mask, mat, dst_size, flags=cv2.INTER_AREA, border_mode=cv2.BORDER_REFLECT_101)
        else:
            rotated_mask = None

        # random channel dropout
        if np.random.rand() < 0.5:
            bgr = cv2.split(rotated)
            bgr[np.random.randint(3)] *= np.random.rand()
            rotated = cv2.merge(bgr)

        # random hue shift
        hsv = cv2.split(cv2.cvtColor(rotated.astype(np.float32)/255., cv2.COLOR_BGR2HSV))
        hsv[0] += np.random.rand() * 360
        n = (hsv[0] / 360).astype(np.uint32)
        hsv[0] -= n * 360  # making hue to be within 0..360

        # randomizing saturation
        hsv[1] *= 1 + (np.random.rand() - 0.5) * 2 * self.max_saturation_delta
        hsv[1] = np.clip(hsv[1], 0, 1)

        # randomizing lightness
        hsv[2] *= 1 + (np.random.rand() - 0.5) * 2 * self.max_lightness_delta
        hsv[2] = np.clip(hsv[2], 0, 1)

        rotated = (cv2.cvtColor(cv2.merge(hsv), cv2.COLOR_HSV2BGR) * 255).astype(np.uint8)

        return rotated, rotated_mask

    def generate_image(self, img, mask, bg, dst_size):
        img, mask = self.transform_image(img, dst_size, mask)
        bg, _ = self.transform_image(bg, dst_size)
        return self.render(img, mask, bg)

    def get_steps_per_epoch(self):
        return int(len(self.image_tuples) / self.batch_size)

    def generate(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            img_indexes = np.arange(len(self.image_tuples))
            np.random.shuffle(img_indexes)
            bg_indexes = np.arange(len(self.bg_images))
            np.random.shuffle(bg_indexes)

            # Generate batches
            bg_index = 0
            img_index = 0
            n = self.get_steps_per_epoch()
            for i in range(n):

                batch = []
                batch_labels = []

                for b in range(self.batch_size):

                    _img_index = img_indexes[img_index]
                    _bg_index = bg_indexes[bg_index]

                    img, mask, label = self.image_tuples[_img_index]
                    bg_img = self.bg_images[_bg_index]

                    generated_img = self.generate_image(img, mask, bg_img, self.image_size)

                    # formatting data for the network
                    batch.append(cv2.split(generated_img))
                    batch_labels.append(label)

                    batch.append(generated_img)

                    bg_index = (bg_index+1) % len(self.bg_images)
                    img_index += 1

                # converting to numpy arrays
                batch = np.array(batch, dtype=np.float32)
                batch_labels = to_categorical(np.array(batch_labels, dtype=np.float32), num_classes=self.num_classes)
                yield batch, batch_labels
