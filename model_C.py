from keras import applications
from keras.engine import Model
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout


class Model_C(object):

    def __init__(self) -> None:
        super().__init__()

    def create_model(self, image_width, image_height, num_classes):
        model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(image_width, image_height, 3))

        x = model.output
        x = Flatten()(x)

        x = Dropout(0.7)(x)

        x = Dense(num_classes)(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

        predictions = x
        return Model(model.input, predictions)
