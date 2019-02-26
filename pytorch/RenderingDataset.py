import torch

import cv2
from torch.utils.data import Dataset

import numpy as np

import ImageDataset
from ImageUtils import convert_to_8bpp


class RenderingDataset(Dataset):

    def __init__(self, backgrounds: ImageDataset, foregrounds: ImageDataset, target_size: tuple):
        super(RenderingDataset, self).__init__()
        self.backgrounds = backgrounds
        self.foregrounds = foregrounds
        self.target_size = target_size
        self.desaturate_prob = 0.7
        self.blur_prob = 0.2
        self.blur_kernel = 5
        self.negative_prob = 0.2
        self.color_ampl = 0.3
        self.color_val_ampl = 0.3
        self.noise_var = 4
        self.max_shadow_intensity = 0.5

    def __len__(self):
        # counting only foregrounds
        return len(self.foregrounds)

    def __getitem__(self, index):
        fg_index = index
        bg_index = index % len(self.backgrounds)

        # fetching foreground and background images
        fg_img, label = self.foregrounds[fg_index]
        bg_img, _ = self.backgrounds[bg_index]

        # add alpha channel to foreground
        fg_img = self.add_alpha(fg_img)

        # random cropping background
        bg_height = np.random.randint(bg_img.shape[0] // 2, bg_img.shape[0])
        bg_width = np.random.randint(bg_img.shape[1] // 2, bg_img.shape[1])
        x0 = self.safe_random(0, bg_img.shape[1] - bg_width)
        y0 = self.safe_random(0, bg_img.shape[0] - bg_height)

        bg_crop = bg_img[y0: y0 + bg_height, x0: x0 + bg_width]
        bg_img = cv2.resize(bg_crop, self.target_size, interpolation=cv2.INTER_AREA)

        # random squeezing foreground
        fg_width = fg_img.shape[1]
        fg_height = fg_img.shape[0]

        random_w_squeeze = 1 - np.random.rand() * 0.2
        random_h_squeeze = 1 - np.random.rand() * 0.2
        tgt_fg_width = min(self.target_size[0], int(fg_width * random_w_squeeze))
        tgt_fg_height = min(self.target_size[1], int(fg_height * random_h_squeeze))

        fg_img = cv2.resize(fg_img, (tgt_fg_width, tgt_fg_height), interpolation=cv2.INTER_AREA)

        # augmentation
        fg_img = self.augment(fg_img)
        bg_img = self.augment(bg_img)

        rgb_channels = self.blend(fg_img, bg_img)
        self.add_random_noise(rgb_channels)

        rgb_tensor = torch.tensor(rgb_channels) / 255.

        # getting label index
        label_index = self.foregrounds.labels.index(label)

        label_tensor = torch.LongTensor([label_index])

        return {'rgb': rgb_tensor, 'label': label_tensor}

    def augment(self, img: np.ndarray) -> np.ndarray:
        """
        Perform augmentation. Tweak colors, add blur, etc
        :param img: image to tweak
        :return: tweaked image
        """

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)

        if np.random.rand() < self.desaturate_prob:
            self.desaturate(img)

        if np.random.rand() < self.blur_prob:
            img = cv2.blur(img, (self.blur_kernel, self.blur_kernel))

        img = self.modulate_colors(img, self.color_ampl, self.color_val_ampl)
        return img

    def modulate_colors(self, img, color_amplitude, value_amplitude):
        img_f = img.astype(np.float32)
        channels = cv2.split(img_f)

        r_scale = 1 + 2 * (np.random.rand() - 0.5) * color_amplitude
        g_scale = 1 + 2 * (np.random.rand() - 0.5) * color_amplitude
        b_scale = 1 + 2 * (np.random.rand() - 0.5) * color_amplitude
        v_scale = 1 + 2 * (np.random.rand() - 0.5) * value_amplitude
        r_scale *= v_scale
        g_scale *= v_scale
        b_scale *= v_scale

        channels[0] *= b_scale
        channels[1] *= g_scale
        channels[2] *= r_scale

        # inverting colors
        for c in range(3):
            if np.random.rand() < self.negative_prob:
                channels[c] = 255 - channels[c]

        return convert_to_8bpp(cv2.merge(channels))

    @staticmethod
    def desaturate(_rgb: np.ndarray) -> np.ndarray:
        """
        Randomly decrease color saturation
        :param _rgb: input image
        :return: desaturated image
        """

        level = np.random.rand()
        rgb = cv2.split(_rgb)
        if _rgb.shape[2] == 3:
            gray = cv2.cvtColor(_rgb, cv2.COLOR_BGR2GRAY) * level
        else:
            gray = cv2.cvtColor(cv2.merge(rgb[0:3]), cv2.COLOR_BGR2GRAY) * level

        for i in range(3):
            rgb[i] = (rgb[i] * (1 - level) + gray).astype(np.uint8)
        return cv2.merge(rgb)

    @staticmethod
    def safe_random(low, high):
        if high > low:
            return np.random.randint(low, high)
        return low

    @staticmethod
    def add_alpha(fg_img):
        channels = cv2.split(fg_img)
        alpha = (channels[0] != 0) | (channels[1] != 0) | (channels[2] != 0)
        alpha = alpha.astype(np.uint8) * 255
        channels.append(alpha)
        return cv2.merge(channels)

    def blend(self, fg: np.ndarray, bg: np.ndarray):

        # check
        if len(fg.shape) != 3 or fg.shape[2] != 4:
            raise Exception("Foreground has to be 4-channel image")

        if len(bg.shape) != 3 or bg.shape[2] != 3:
            raise Exception("Background has to be 3-channel image")

        # positioning foreground randomly
        diff_x = bg.shape[1] - fg.shape[1]
        diff_y = bg.shape[0] - fg.shape[0]

        x0 = RenderingDataset.safe_random(0, diff_x)
        y0 = RenderingDataset.safe_random(0, diff_y)

        _fg = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
        _fg[y0: y0 + fg.shape[0], x0: x0 + fg.shape[1]] = fg
        fg = _fg

        # calculating shadow
        alpha = cv2.split(fg)[3]
        shadow_intensity = np.random.rand() * self.max_shadow_intensity
        shadow = (255 - cv2.blur(alpha, (31, 31)).astype(np.float32) * shadow_intensity) / 255.

        bgf = bg.astype(np.float32)
        fgf = fg.astype(np.float32)
        fg_channels = cv2.split(fgf)
        alpha = fg_channels[3]
        alpha_inv = 255 - alpha

        # applying shadow
        bg_channels = cv2.split(bgf)
        for c in range(3):
            bg_channels[c] *= shadow

        for c in range(3):
            bg_channels[c] = (fg_channels[c] * alpha + bg_channels[c] * alpha_inv) / 255

        return bg_channels

    def add_random_noise(self, rgb_channels):
        for c in range(len(rgb_channels)):
            noise_matrix = np.random.normal(0, self.noise_var, size=rgb_channels[c].shape)
            rgb_channels[c] += noise_matrix

