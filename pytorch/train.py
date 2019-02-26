import os

import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader

from DebugDumper import DebugDumper
from ImageDataset import ImageDataset
from PlotUtils import plot_histograms, plot_train_curves
from RenderingDataset import RenderingDataset
from modelA import modelA


def train(data_loader: DataLoader, model: nn.Module, num_iterations: int, start_iteration: int, device: torch.device,
          losses_dict: dict, accuracies_dict: dict, optimizer: torch.optim.Optimizer = None,
          debug_dumper: DebugDumper = None):

    loss_func = nn.CrossEntropyLoss()

    iterator = iter(data_loader)
    epoch_size = len(data_loader.dataset)

    is_train = optimizer is not None

    i = 0
    global_av_loss = 0
    global_av_accuracy = 0
    for i in range(num_iterations):

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)

        # debug dump
        if debug_dumper is not None:
            debug_dumper.dump(batch)

        # transferring data to device
        for k in batch:
            batch[k] = batch[k].to(device)

        av_loss = 0
        av_accuracy = 0
        av_n = 0
        with torch.set_grad_enabled(is_train):

            predictions = model.forward(batch)

            labels = batch['label']
            labels = labels.view(labels.size(0))
            loss = loss_func(predictions, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            av_loss += loss.item()
            global_av_loss += loss.item()

            # calculating accuracy
            predicted_labels = torch.argmax(predictions, dim=1)
            correct_count = predicted_labels.eq(labels).sum().item()
            accuracy = correct_count / labels.size(0)
            av_accuracy += accuracy
            global_av_accuracy += accuracy

            av_n += 1

        iteration = i + start_iteration
        if i % 10 == 0:
            e = iteration / epoch_size
            av_loss /= av_n
            av_accuracy /= av_n
            if is_train:
                print("%d: epoch=%1.2f, loss=%f, accuracy=%1.2f%%" % (iteration, e, av_loss, av_accuracy*100))
            else:
                print("%d: loss=%f, accuracy=%1.2f%%" % (i, av_loss, av_accuracy*100))

        # adding loss to log
        if is_train:
            losses_dict[iteration] = loss.item()
            accuracies_dict[iteration] = accuracy

    if not is_train:
        global_av_loss /= num_iterations
        global_av_accuracy /= num_iterations
        print("average test loss = %f, average test accuracy = %f" % (global_av_loss, global_av_accuracy))
        losses_dict[num_iterations + start_iteration] = global_av_loss
        accuracies_dict[num_iterations + start_iteration] = global_av_accuracy

    return num_iterations + start_iteration


def main(args):

    batch_size = 32
    snapshot_iters = 500
    test_iters = 100
    snapshot_dir = "snapshots"
    image_size = (224, 224)
    learning_rate = 0.01
    debug_dir = "debug"
    debug_number = 0

    use_cuda = torch.cuda.is_available() and args.gpu.lower() == 'true'
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using GPU" if use_cuda else "Using CPU")

    # create data sets
    train_foregrounds = ImageDataset(os.path.join(args.data_root, "train.txt"), "sorted")
    test_foregrounds = ImageDataset(os.path.join(args.data_root, "val.txt"), "sorted")
    num_classes = len(train_foregrounds.labels)
    backgrounds = ImageDataset(os.path.join(args.data_root, "backgrounds"), forceRgb=True)

    print("Number of classes = %d" % num_classes)
    print("Number of train foregrounds = %d" % len(train_foregrounds))
    print("Number of test foregrounds = %d" % len(test_foregrounds))
    print("Number of backgrounds = %d" % len(backgrounds))

    if len(train_foregrounds) == 0 or len(test_foregrounds) == 0 or len(backgrounds) == 0:
        raise Exception("One of datasets is empty")

    train_dataset = RenderingDataset(backgrounds, train_foregrounds, image_size)
    test_dataset = RenderingDataset(backgrounds, test_foregrounds, image_size)

    # create data loaders
    kwargs = {'num_workers': 6, 'pin_memory': False} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # creating model
    model = modelA(num_classes, False).to(device)
    model_name = type(model).__name__

    # preparing snapshot dir
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)

    # preparing model dir
    model_dir = os.path.join(snapshot_dir, model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    iteration = 0
    if args.snapshot is not None:
        iteration = args.snapshot
        snapshot_file = get_snapshot_file_name(iteration, model_dir)
        print("loading " + snapshot_file)
        model.load_state_dict(torch.load(snapshot_file))

    print('Starting from iteration %d' % iteration)

    # creating optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # creating logs
    train_losses = {}
    train_accuracies = {}
    test_losses = {}
    test_accuracies = {}

    # creating debug dumper
    debug_dumper = DebugDumper(debug_dir, debug_number)

    while iteration < args.num_iterations:

        # train
        print("training...")
        iteration = train(train_loader, model, snapshot_iters, iteration, device, train_losses, train_accuracies, optimizer, debug_dumper)

        # dumping snapshot
        snapshot_file = get_snapshot_file_name(iteration, model_dir)
        print("dumping snapshot: " + snapshot_file)
        torch.save(model.state_dict(), snapshot_file)

        # test
        print("testing...")
        train(test_loader, model, test_iters, iteration, device, test_losses, test_accuracies)

        # visualizing training progress
        plot_histograms(model, model_dir)
        plot_train_curves(train_losses, test_losses, "TrainCurves", model_dir)
        plot_train_curves(train_accuracies, test_accuracies, "Accuracy", model_dir)


def get_snapshot_file_name(iteration, model_dir):
    snapshot_file = os.path.join(model_dir, "snapshot_" + str(iteration) + ".pt")
    return snapshot_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training and test after every N iterations.')
    parser.add_argument("model", type=str, default=None, help="Name of the model to train.")
    parser.add_argument("--data_root", type=str, default=None, help="Path to data.")
    parser.add_argument("--snapshot", type=int, default=None, help="Iteration to start from snapshot.")
    parser.add_argument("--num_iterations", type=int, default=100000, help="Number of iterations to train.")
    parser.add_argument("--gpu", type=str, default='true', help="Set to 'true' for GPU.")
    _args = parser.parse_args()
    main(_args)
