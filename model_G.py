from keras import applications
from keras.engine import Model
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout


class Model_G(object):

    def __init__(self) -> None:
        super().__init__()

    def create_model(self, image_width, image_height, num_classes):
        model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(image_width, image_height, 3))

        x = model.output
        x = Flatten()(x)

        dropout = Dropout(0.7)(x)

        x = Dense(num_classes)(dropout)
        x = BatchNormalization()(x)
        predictions = Activation('softmax', name="classes")(x)

        dimensions = Dense(2, name="dimensions")(dropout)

        return Model(model.input, outputs=[predictions, dimensions])
