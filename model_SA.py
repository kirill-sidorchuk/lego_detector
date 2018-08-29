from keras import applications
from keras.engine import Model
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Conv2DTranspose, concatenate, Convolution2D, \
    Cropping2D, ZeroPadding2D


class Model_SA(object):

    def __init__(self) -> None:
        super().__init__()

    def create_model(self, image_width, image_height, num_classes):
        model = applications.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_width, image_height, 3))

        # freezing all layers
        # for layer in model.layers:
        #     layer.trainable = False

        # input: (None, 224, 224, 3)

        # segmentation branch
        out_7_7 = model.get_layer('activation_49').output    # (?, 7, 7, 2048)
        out_14_14 = model.get_layer('activation_40').output  # (?, 14, 14, 1024)
        out_28_28 = model.get_layer('activation_22').output  # (?, 28, 28, 512)
        out_55_55 = model.get_layer('activation_10').output  # (?, 55, 55, 256)

        deconv7 = Conv2DTranspose(512, kernel_size=2, strides=2, padding="valid", output_padding=0)(out_7_7)
        deconv7_bn = BatchNormalization()(deconv7)
        deconv7_relu = Activation("relu")(deconv7_bn)
        # (?, 14, 14, 512)

        concat14 = concatenate([out_14_14, deconv7_relu])
        # (?, 14, 14, 1024+512)

        proj14 = Convolution2D(512, kernel_size=1, strides=1, padding="valid")(concat14)
        proj14_bn = BatchNormalization()(proj14)
        proj14_relu = Activation("relu")(proj14_bn)
        # (?, 14, 14, 512)

        deconv14 = Conv2DTranspose(256, kernel_size=2, strides=2, padding="valid", output_padding=0)(proj14_relu)
        deconv14_bn = BatchNormalization()(deconv14)
        deconv14_relu = Activation("relu")(deconv14_bn)
        # (?, 28, 28, 256)

        concat28 = concatenate([out_28_28, deconv14_relu])
        # (?, 28, 28, 512+256)

        proj28 = Convolution2D(256, kernel_size=1, strides=1, padding="valid")(concat28)
        proj28_bn = BatchNormalization()(proj28)
        proj28_relu = Activation("relu")(proj28_bn)
        # (?, 28, 28, 256)

        deconv28 = Conv2DTranspose(128, kernel_size=2, strides=2, padding="valid", output_padding=0)(proj28_relu)
        deconv28_bn = BatchNormalization()(deconv28)
        deconv28_relu = Activation("relu")(deconv28_bn)
        # (?, 56, 56, 256)

        # use cropping to get from 56 to 55
        crop55 = Cropping2D(((1,0), (0,1)))(deconv28_relu)

        concat55 = concatenate([out_55_55, crop55])
        # (?, 55, 55, 256+128)

        proj55 = Convolution2D(64, kernel_size=1, strides=1, padding="valid")(concat55)
        proj55_bn = BatchNormalization()(proj55)
        proj55_relu = Activation("relu")(proj55_bn)
        # (?, 55, 55, 64)

        # upsampling to original image resolution
        deconv55 = Conv2DTranspose(2, kernel_size=4, strides=4, padding="valid")(proj55_relu)
        # (?, 220, 220, 2)

        # add padding to get 224x224
        zeropad55 = ZeroPadding2D(padding=2)(deconv55)
        # (?, 224, 224, 2)

        segm_activation = Activation('softmax', name="segm_out")(zeropad55)


        # class prediction branch
        x = model.output

        x = Dropout(0.8)(x)

        x = Dense(num_classes)(x)
        x = BatchNormalization()(x)
        x = Activation('softmax', name="class_out")(x)

        predictions = [x, segm_activation]
        return Model(model.input, predictions)
