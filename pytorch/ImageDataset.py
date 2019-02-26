import os

import cv2
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset for foregrounds and backgrounds
    """
    img_exts = {'.png', '.jpg'}

    def __init__(self, root: str, subdir: str = None, forceRgb = False):
        super(ImageDataset, self).__init__()
        if os.path.isfile(root):
            self.images = self._load_list(root, subdir)
            self.labels = self._get_labels(self.images, os.path.dirname(root))
        else:
            self.images = self._search_for_files(root)
        self.forceRgb = forceRgb

    def __getitem__(self, index):
        filename = self.images[index]
        label = self._get_label(filename)

        img = cv2.imread(filename, cv2.IMREAD_COLOR if self.forceRgb else cv2.IMREAD_UNCHANGED)

        return img, label

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_list(list_file: str, subdir: str) -> list:
        """
        Load and parse filename with list of images
        :param list_file: list file name
        :param subdir: subdir
        :return: list of images filenames
        """

        with open(list_file, "r") as f:
            lines = f.readlines()

        root = os.path.split(list_file)[0]
        imgs = []
        for l in lines:
            l = l.strip()
            if len(l) < 1:
                continue
            imgs.append(os.path.join(root, subdir, l))
        return imgs

    def _search_for_files(self, root_dir: str) -> list:
        """
        Search for all image files in a directory. Files with extensions in self.img_exts are collected.
        :param root_dir: directory to search in
        :return: list of paths to images
        """

        files = os.listdir(root_dir)
        imgs = []
        for f in files:
            path = os.path.join(root_dir, f)
            if os.path.isfile(path):
                ext = os.path.splitext(f)[1].lower()
                if ext in self.img_exts:
                    imgs.append(path)
        return imgs

    @staticmethod
    def _get_label(filename: str) -> str:
        """
        Parse filename and extract label, which is the name of parent directory
        :param filename: source filename (full path)
        :return: label
        """

        label = os.path.split(os.path.dirname(filename))[1]
        return label

    @staticmethod
    def _get_labels(images: list, root: str) -> list:
        """
        Count all labels found in list of images
        :param images: list of source images (list of filenames)
        :return: count of labels
        """

        labels = set()
        for filename in images:
            labels.add(ImageDataset._get_label(filename))

        # loading saved list
        list_filename = os.path.join(root, "labels.txt")
        saved_labels = None
        if os.path.exists(list_filename):
            with open(list_filename, "r") as f:
                saved_labels = f.readlines()

            saved_labels = [s.strip() for s in saved_labels]

            if len(saved_labels) != len(labels):
                print("WARNING: saved labels size (%d) does not match current count of labels (%d)" %
                      (len(saved_labels), len(labels)))
                saved_labels = None

        if saved_labels is not None:
            # checking if all labels are there
            for saved_label in saved_labels:
                if saved_label not in labels:
                    print("WARNING: saved labels are not the same as current labels: extra label = %s" % saved_label)
                    saved_labels = None
                    break

        if saved_labels is None:
            saved_labels = list(labels)

            # saving to file
            with open(list_filename, "w") as f:
                for l in saved_labels:
                    f.write(l + '\n')

        return saved_labels
