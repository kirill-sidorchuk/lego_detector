import argparse
import numpy as np
import os
from keras import applications
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, CSVLogger

from DataGenerator import DataGenerator

SNAPSHOTS_PATH = "snapshots"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 16

np.random.seed(1337)  # for reproducibility


def parse_epoch(snapshot):
    i = snapshot.find('-')
    if i == -1:
        return 0
    j = snapshot.find('-', i+1)
    if j == -1:
        return 0
    return int(snapshot[i+1:j])


def create_model(num_classes):
    model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # model.summary()

    # freezing all layers
    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(model.input, predictions)


def finetune(args):
    train_data_generator = DataGenerator(args.data_root, "train", 100, 2, 1.1, 1.1, BATCH_SIZE, (IMAGE_HEIGHT, IMAGE_WIDTH))
    val_data_generator = DataGenerator(args.data_root, "val", 100, 2, 1.1, 1.1, BATCH_SIZE, (IMAGE_HEIGHT, IMAGE_WIDTH))

    num_classes = train_data_generator.get_num_classes()

    # creating model
    model_name = "A"
    model = create_model(num_classes)

    # preparing directories for snapshots
    if not os.path.exists(SNAPSHOTS_PATH):
        os.mkdir(SNAPSHOTS_PATH)

    model_snapshot_path = os.path.join(SNAPSHOTS_PATH, model_name)
    if not os.path.exists(model_snapshot_path):
        os.mkdir(model_snapshot_path)

    start_epoch = 0
    if args.snapshot is not None:
        start_epoch = parse_epoch(args.snapshot)
        print("loading weights from epoch %d" % start_epoch)
        model.load_weights(os.path.join(model_snapshot_path, args.snapshot), by_name=True)

    # print summary
    model.summary()

    nb_epoch = 100
    sgd = optimizers.Adam(lr=0.01, decay=1e-4, beta_1=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    filepath = os.path.join(model_snapshot_path, "weights-{epoch:03d}-{val_acc:.3f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    logpath = "model_" + model_name + "_log.txt"
    csv_logger = CSVLogger(logpath)

    # log dir for tensorboard
    tb_log_dir = "model_" + model_name + "_tb"
    if os.path.exists(tb_log_dir):
        # clear_dir(tb_log_dir)
        pass
    else:
        os.mkdir(tb_log_dir)

    tb_log = TensorBoard(tb_log_dir)

    callbacks_list = [checkpoint, csv_logger, tb_log]

    model.fit_generator(generator=train_data_generator.generate(), steps_per_epoch=train_data_generator.get_steps_per_epoch(),
                        epochs=nb_epoch, callbacks=callbacks_list,
                        validation_data=val_data_generator.generate(), validation_steps=val_data_generator.get_steps_per_epoch(),
                        initial_epoch=start_epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("--snapshot", type=str, help="restart from snapshot")

    _args = parser.parse_args()
    finetune(_args)