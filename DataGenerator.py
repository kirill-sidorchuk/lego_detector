import os
import numpy as np
import cv2
#from keras.utils import to_categorical

import ImageUtils
from FilesAndDirs import clear_directory
from FinalizeDataset import SORTED_DIR

BG_RESOLUTION = 1024


class DataGenerator(object):
    def __init__(self, data_root, dataset_type, max_rotation, max_img_zoom, max_bg_zoom, max_saturation_delta, max_lightness_delta,
                 additive_noise, batch_size, image_size, debug_epochs,
                 aug_flip, aug_channel_dropout, aug_hue):
        self.data_root = data_root
        self.max_rotation = max_rotation
        self.max_img_zoom = max_img_zoom
        self.max_bg_zoom = max_bg_zoom
        self.max_saturation_delta = max_saturation_delta
        self.max_lightness_delta = max_lightness_delta
        self.additive_noise = additive_noise
        self.image_size = image_size
        self.batch_size = batch_size
        self.debug_epochs = debug_epochs
        self.aug_flip = aug_flip
        self.aug_channel_dropout = aug_channel_dropout
        self.aug_hue = aug_hue

        if self.debug_epochs:
            self.prepare_debug_dir()

        self.image_files, self.labels_to_ints = self.load_labels(dataset_type + ".txt")
        self.images = self.load_images()
        self.num_classes = len(self.images)
        self.image_tuples = self.initialize_image_order()

        self.bg_images = self.load_backgrounds()

    def dump_labels_to_int_mapping(self, filename):
        with open(filename, "w") as f:
            for label in self.labels_to_ints:
                index = self.labels_to_ints[label]
                f.write("%s, %d\n" % (label, index))

    def prepare_debug_dir(self):
        self.debug_dir = os.path.join(self.data_root, "debug")
        if not os.path.exists(self.debug_dir):
            os.mkdir(self.debug_dir)
        else:
            clear_directory(self.debug_dir)

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

        labels_to_ints = {}
        label_index = 0
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
                labels_to_ints[label] = label_index
                label_index += 1

        return images, labels_to_ints

    def load_images(self):
        images = {}
        max_img_size = (max(self.image_size)*2)//3
        for _class in self.image_files:
            imgs = self.image_files[_class]
            for img_file in imgs:
                # load image
                img = cv2.imread(img_file)
                if img is None:
                    raise Exception("cannot read " + img_file)

                if max(img.shape) > max_img_size:
                    img = ImageUtils.resize_to_resolution(img, max_img_size)

                # create mask
                img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_mask = (img_bw != 0).astype(np.uint8)

                if _class in images:
                    images[_class].append((img, img_mask))
                else:
                    images[_class] = [(img, img_mask)]

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

        # shifting foreground
        max_shift_x = bg_image.shape[1] - fg_image.shape[1]
        max_shift_y = bg_image.shape[0] - fg_image.shape[0]
        fg_big = np.zeros(bg_image.shape, dtype=np.uint8)
        mask_big = np.zeros((bg_image.shape[0], bg_image.shape[1]), dtype=np.uint8)

        shift_x = np.random.randint(0, max_shift_x)
        shift_y = np.random.randint(0, max_shift_y)

        fg_big[shift_y:shift_y+fg_image.shape[0], shift_x:shift_x+fg_image.shape[1], :] = fg_image
        mask_big[shift_y:shift_y+fg_image.shape[0], shift_x:shift_x+fg_image.shape[1]] = fg_mask

        dst[mask_big != 0] = 0
        dst += fg_big
        return dst

    def get_num_classes(self):
        return self.num_classes

    def transform_image(self, img, dst_size, mask=None):

        # random rotation
        if self.max_rotation != 0:
            img, mask = self.random_rotation(dst_size, img, mask)

        if self.aug_flip:
            img, mask = self.random_flip(img, mask)

        if self.aug_channel_dropout:
            img = self.random_channel_dropout(img)

        # random hue shift
        if self.aug_hue:
            img = self.random_hue_shift(img)

        return img, mask

    def random_flip(self, img, mask):
        if np.random.randint(100) < 50:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)

        if np.random.randint(100) < 50:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        return img, mask

    def random_rotation(self, dst_size, img, mask):
        if mask is not None:
            # pasting image into a bigger square matrix to prevent edge clipping
            w = img.shape[1]
            h = img.shape[0]
            big_size = max(w, h) * 2
            big_img = np.zeros((big_size, big_size, 3), dtype=np.uint8)
            big_mask = np.zeros((big_size, big_size), dtype=np.uint8)
            x0 = (big_size - w)//2
            y0 = (big_size - h)//2
            big_img[y0:y0 + h, x0:x0 + w, :] = img
            big_mask[y0:y0 + h, x0:x0 + w] = mask
            img = big_img
            mask = big_mask

        center = (img.shape[1] / 2, img.shape[0] / 2)
        angle = np.random.rand() * self.max_rotation
        if mask is None:
            max_zoom = self.max_bg_zoom
        else:
            max_zoom = self.max_img_zoom

        scale = max_zoom[0] + np.random.rand() * (max_zoom[1] - max_zoom[0])
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        border_mode = cv2.BORDER_REFLECT_101 if mask is None else cv2.BORDER_CONSTANT
        rotated = cv2.warpAffine(img, mat, dst_size, flags=cv2.INTER_AREA, borderMode=border_mode, borderValue=0)
        if mask is not None:
            rotated_mask = cv2.warpAffine(mask, mat, dst_size, flags=cv2.INTER_AREA, borderMode=border_mode,
                                          borderValue=0)
            try:
                nozero_coords = np.where(rotated_mask != 0)
                tl = (min(nozero_coords[1]), min(nozero_coords[0]))
                br = (max(nozero_coords[1]), max(nozero_coords[0]))
                rotated = rotated[tl[1]:br[1], tl[0]:br[0]]
                rotated_mask = rotated_mask[tl[1]:br[1], tl[0]:br[0]]
            except Exception as e:
                pass
        else:
            rotated_mask = None
        return rotated, rotated_mask

    def random_hue_shift(self, img):
        hsv = cv2.split(cv2.cvtColor(img.astype(np.float32) / 255., cv2.COLOR_BGR2HSV))
        hsv[0] += np.random.rand() * 360
        n = (hsv[0] / 360).astype(np.uint32)
        hsv[0] -= n * 360  # making hue to be within 0..360
        # randomizing saturation
        hsv[1] *= 1 + (np.random.rand() - 0.5) * 2 * self.max_saturation_delta
        hsv[1] = np.clip(hsv[1], 0, 1)
        # randomizing lightness
        hsv[2] *= 1 + (np.random.rand() - 0.5) * 2 * self.max_lightness_delta
        hsv[2] = np.clip(hsv[2], 0, 1)
        img = (cv2.cvtColor(cv2.merge(hsv), cv2.COLOR_HSV2BGR) * 255).astype(np.uint8)
        return img

    def random_channel_dropout(self, img):
        if np.random.rand() < 0.5:
            bgr = cv2.split(img)
            channel = np.random.randint(3)
            bgr[channel] = (bgr[channel] * (0.1 + 0.8 * np.random.rand())).astype(np.uint8)
            img = cv2.merge(bgr)

        return img

    def generate_image(self, img, mask, bg, dst_size):
        img, mask = self.transform_image(img, dst_size, mask)
        bg, _ = self.transform_image(bg, dst_size)
        final_img = self.render(img, mask, bg)
        final_img = np.clip(final_img.astype(np.int32) + \
                    (np.random.randn(final_img.shape[0], final_img.shape[1], final_img.shape[2]) * self.additive_noise).astype(np.int32),
                    0, 255).astype(np.uint8)
        return final_img

    def get_steps_per_epoch(self):
        return int(len(self.image_tuples) / self.batch_size)

    def generate(self):
        'Generates batches of samples'

        debug_epoch = self.debug_epochs
        epoch = 0

        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            img_indexes = np.arange(len(self.image_tuples))
            np.random.shuffle(img_indexes)
            bg_indexes = np.arange(len(self.bg_images))
            np.random.shuffle(bg_indexes)

            epoch += 1
            debug_epoch -= 1

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
                    if debug_epoch >= 0:
                        # saving generated image
                        filename = os.path.join(self.debug_dir, "%02d_%03d.jpg" % (epoch, b))
                        cv2.imwrite(filename, generated_img)

                    # formatting data for the network
                    batch.append(generated_img)
                    batch_labels.append(self.labels_to_ints[label])

                    bg_index = (bg_index + 1) % len(self.bg_images)
                    img_index += 1

                # converting to numpy arrays
                batch = np.array(batch, dtype=np.float32) / 255.0
                batch_labels = 0 #to_categorical(np.array(batch_labels, dtype=np.float32), num_classes=self.num_classes)
                yield batch, batch_labels
