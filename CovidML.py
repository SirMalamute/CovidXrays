###########Building the model
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def alexnet(in_shape=(256,256,1), n_classes=3, opt='sgd'):
    in_layer = layers.Input(in_shape)
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)
    pool1 = layers.MaxPool2D(3, 2)(conv1)
    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D(3, 2)(conv2)
    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D(3, 2)(conv4)
    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(4096, activation='relu')(flattened)
    drop1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(4096, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(dense2)
    preds = layers.Dense(n_classes, activation='softmax')(drop2)

    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
	              metrics=["accuracy"])
    return model


####Flowing the data
train_path = './Datasets/Train'
validation_path = './Datasets/Validation'

folders = glob('./Datasets/Train/*')

train_data_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.1)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_set = train_data_gen.flow_from_directory(train_path, target_size=(256,256), batch_size=32, class_mode='categorical')
test_set = test_data_gen.flow_from_directory(validation_path, target_size=(256,256), batch_size=32, class_mode='categorical')

#### Instantiating the model
model = alexnet()
model.summary()
