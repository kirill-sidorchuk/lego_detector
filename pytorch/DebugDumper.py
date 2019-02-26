import os

import cv2
import torch

from ImageUtils import convert_to_8bpp


class DebugDumper:
    """
    Class to save images sent to the network. For debug purposes.
    """

    def __init__(self, debug_dir: str, count: int):
        self.debug_dir = debug_dir
        self.count = count

        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

    def dump(self, batch: dict) -> None:
        """
        Dumps all stuff from batch which is recognized.
        :param batch: dictionary
        :return:
        """

        if self.count <= 0:
            return

        for key in batch:
            if key == 'rgb':
                self.count -= self.dump_rgb(batch[key], self.count)

    def dump_rgb(self, tensor: torch.Tensor, count: int) -> int:
        """
        Dump RGB images
        :param tensor: tensor for image
        :param count: counter
        :return: new counter
        """

        for i in range(len(tensor)):

            img = convert_to_8bpp(cv2.merge(tensor[i].numpy() * 255))
            file_name = os.path.join(self.debug_dir, "%d.jpg" % (count-i))
            cv2.imwrite(file_name, img)

        return count - len(tensor)