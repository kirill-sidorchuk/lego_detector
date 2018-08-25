from keras import applications
from keras.engine import Model
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout


class Model_A(object):

    def __init__(self) -> None:
        super().__init__()

    def create_model(self, image_width, image_height, num_classes):
        model = applications.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_width, image_height, 3))

        # freezing all layers
        # for layer in model.layers:
        #     layer.trainable = False

        x = model.output
        # x = Flatten()(x) # not needed anymore?

        x = Dropout(0.8)(x)

        x = Dense(num_classes)(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)

        predictions = x
        return Model(model.input, predictions)
