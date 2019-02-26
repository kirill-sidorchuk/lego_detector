import math
import os

import torch
import matplotlib

from GradientCollector import SQUARED_SUFFIX, RELATIVE_SQUARED_SUFFIX

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch.nn as nn


def plot_histograms(model: nn.Module, dir: str):

    if dir is not None and not os.path.exists(dir):
        os.mkdir(dir)

    params = []
    for name, value in model.named_parameters():
        if 'weight' in name and value.requires_grad:
            params.append((name, value))

    for (name, value) in params:
        v = value.detach().cpu().numpy()
        plt.hist(v.flatten(), 50)
        plt.title(name)

        if dir is not None:
            plt.savefig(os.path.join(dir, "hist_" + name + ".png"))
            plt.clf()
        else:
            plt.show()

    # visualizing first conv layer
    if hasattr(model, 'get_first_conv_layer'):
        first_conv_weights = model.get_first_conv_layer().weight
        img = visualize_conv_layer(first_conv_weights)
        if dir is not None:
            cv2.imwrite(os.path.join(dir, 'first_conv.png'), img)
        else:
            plt.imshow(img)
            plt.clf()


def split_dict(vals_dict: dict) -> tuple:
    """
    Split and sort keys and values
    :param vals_dict:
    :return:
    """

    keys = [key for key in vals_dict]
    keys.sort()
    values = [vals_dict[iter] for iter in keys]

    return keys, values


def plot_gradients(grads: dict, title: str, dir: str) -> None:
    """
    Plotting average absolute values of gradients for all layers
    :param grads: collected gradients, returned by train()
    :param title: title for the plot
    :param dir: directory to save image
    """

    # extracting two sorted arrays out of dictionary
    iters, _grads = split_dict(grads)

    x0 = iters[0]
    x1 = iters[-1]

    for layer in _grads[0]:

        if layer.endswith(SQUARED_SUFFIX):
            value_type = "gradients"
        elif layer.endswith(RELATIVE_SQUARED_SUFFIX):
            value_type = "grad_relative"
        else:
            continue

        axes = plt.gca()
        axes.set_yscale("log")

        mean_grads = []
        for iter in iters:
            g = grads[iter][layer]
            avg = math.sqrt(np.mean(g.flatten()))
            mean_grads.append(avg)

        plt.scatter(iters, mean_grads, c='r', alpha=0.25)
        smoothed_data = smooth_data(mean_grads, 1 + int((x1-x0) * 0.01))
        plt.plot(iters, smoothed_data, '-b', label="train")
        plt.title(title + " " + value_type + " " + layer)

        if dir is not None:
            if not os.path.exists(dir):
                os.mkdir(dir)

            plot_file = os.path.join(dir, "%s_%s_%s.png" % (value_type, title, layer))
            plt.savefig(plot_file)
        else:
            plt.show()

        plt.clf()
        plt.close()


def plot_train_curves(_train_losses: dict, _test_losses: dict, title: str, dir: str, only_test: bool = False):

    # extracting two sorted arrays out of losses dictionaries
    train_iters, train_losses = split_dict(_train_losses)
    test_iters, test_losses = split_dict(_test_losses)

    axes = plt.gca()
    axes.set_yscale("log")

    if not only_test:
        plt.scatter(train_iters, train_losses, c='r', alpha=0.25)

        x0 = train_iters[0]
        x1 = train_iters[-1]
        smoothed_data = smooth_data(train_losses, 1 + int((x1-x0) * 0.01))
        plt.plot(train_iters, smoothed_data, '-b', label="train")

    plt.plot(test_iters, test_losses, color='g', marker='o', linewidth=2)
    plt.title(title)

    if dir is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)

        plot_file = os.path.join(dir, "train_curves_%s.png" % title)
        plt.savefig(plot_file)
    else:
        plt.show()

    plt.clf()
    plt.close()


def smooth_data(data: list, kernel_size: int) -> np.ndarray:
    """
    Applies median averaging to data
    :param data: 1D array of floats
    :param kernel_size: averaging kernel size
    :return: numpy array of averaged data
    """
    smoothed = np.ndarray(len(data))
    s2 = int(kernel_size / 2)
    for i in range(len(data)):
        i0 = max(0, i - s2)
        i1 = min(len(data), i + s2)
        av = np.median(data[i0:i1])
        smoothed[i] = av
    return smoothed


def visualize_conv_layer(weights: torch.Tensor) -> np.ndarray:
    """
    Creates visualization of convolutional kernel
    :param weights: torch tensor with neuron's weights
    :return: numpy array with image
    """

    num_neurons = weights.shape[0]

    grid_nx = int(math.sqrt(num_neurons) * 1.5)
    grid_ny = math.ceil(num_neurons / grid_nx)

    image_width = 800
    image_height = 600
    pad = 10
    inner_space = 4

    image = np.ndarray((image_height, image_width, 3), dtype=np.uint8)
    image.fill(255)

    cell_width = (image_width - 2 * pad) / grid_nx
    cell_height = (image_height - 2 * pad) / grid_ny

    cell_inner_width = int(cell_width - inner_space)
    cell_inner_height = int(cell_height - inner_space)

    for j in range(grid_ny):
        cell_y0 = int(pad + cell_height * j)
        for i in range(grid_nx):
            cell_x0 = int(pad + cell_width * i)

            neuron_index = i + j * grid_nx
            if neuron_index >= num_neurons:
                break

            kernel_image = get_kernel_image(weights[neuron_index])
            img_resized = cv2.resize(kernel_image, (cell_inner_width, cell_inner_height), interpolation=cv2.INTER_NEAREST)
            image[cell_y0: cell_y0 + cell_inner_height, cell_x0: cell_x0 + cell_inner_width] = img_resized

    return image


def get_kernel_image(weights) -> np.ndarray:
    """
    Convert neuron kernel weights to RGB image
    :param weights: neuron's weight
    :return: RGB image
    """

    if weights.shape[0] == 1:
        img = cv2.cvtColor(weights[0].detach().cpu().numpy(), cv2.COLOR_GRAY2BGR)
    elif weights.shape[0] == 3:
        img = cv2.merge(weights.detach().cpu().numpy())
    else:
        raise Exception('Unsupported number of channels in neuron: ' + str(weights.shape[0]))

    cv2.normalize(img, img, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return img
