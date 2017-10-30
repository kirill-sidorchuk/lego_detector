from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50

num_classes = 28

model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
model.summary()

# freezing all layers
for layer in model.layers:
    layer.trainable = False

x = model.output
x = Flatten()(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

predictions = Dense(num_classes, activation='softmax')(x)