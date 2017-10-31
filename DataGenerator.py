import os
import numpy as np
import cv2

from FinalizeDataset import SORTED_DIR


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
        self.image_tuples = self.initialize_image_order()

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
        bg_image[fg_mask] = fg_image

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
            indexes = np.arange(len(self.data))
            np.random.shuffle(indexes)

            # Generate batches
            n = self.get_steps_per_epoch()
            for i in range(n):

                batch = []
                batch_labels = []

                for b in range(self.batch_size):
                    # getting random data file
                    raw_data = self.data[np.random.randint(len(self.data))]

                    # getting random sequence from the file
                    seq_i = np.random.randint(len(raw_data) - self.seq_length - 2)
                    sequence = raw_data[seq_i: seq_i + self.seq_length + 1].copy()  # add extra sample at the end as label

                    # normalizing sequence (together with next rate sample)
                    normalize_sequence(sequence)

                    # data augmentation

                    # additive noise
                    sequence += np.random.randn(sequence.shape[0], sequence.shape[1]) * AUG_NOISE_SIGMA

                    # multiplicative noise
                    sequence[:, TIME_COLUMN] *= 1 + np.random.randn() * MULT_NOISE_FOR_TIME
                    sequence[:, RATE_COLUMN] *= 1 + np.random.randn() * MULT_NOISE_FOR_RATE
                    sequence[:, VOLUME_COLUMN] *= 1 + np.random.randn() * MULT_NOISE_FOR_RATE

                    next_rate = sequence[-1, RATE_COLUMN]

                    batch.append(sequence[0:-1])
                    batch_labels.append(next_rate)

                yield np.array(batch, dtype=np.float32), np.array(batch_labels, dtype=np.float32)

