"""# Loading Train, Val, and Test Data Using Keras ImageDataGenerator"""

# Directory Paths
#train_dir = "/gdrive/My Drive/newCircleData/Train/"
#val_dir = "/gdrive/My Drive/newCircleData/Val/"
test_dir = "/hdd/kavi/test/newCircleData/Test/" #"/gdrive/My Drive/mapsRedCap_DCH/fullReports/" 
train_data_dir = "/hdd/kavi/test/newCircleData/TrainValTest/"

import numpy as np 
import tensorflow as tf
#from numpy.random import seed
#tf.random.set_seed(seed(2))

#Image dims and training details
img_width = 448
img_height = 448
batch_size = 1
channels = 3
epochs = 50
#nb_train_samples = 395
#nb_valid_samples = 145
nb_test_samples = 135

#Keras ImageDataGenerator for loading train, val, and test data
from keras.preprocessing.image import ImageDataGenerator

#train_datagen = ImageDataGenerator(rescale=1./255)             
#valid_datagen = ImageDataGenerator(rescale=1./255)    
test_datagen = ImageDataGenerator(rescale=1./255) 

train_datagen = ImageDataGenerator(rescale=1./255,
    #shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    #save_to_dir='/gdrive/My Drive/augmentedInceptionTrain/',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    #save_to_dir='/gdrive/My Drive/augmentedInceptionVal/',
    subset='validation') # set as validation data

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

label_map = train_generator.class_indices
print(label_map)

filenames = test_generator.filenames 
print(filenames)

"""# Building the Model Architecture (Pre-trained DenseNet-121 and Fine-Tuning on OCT Dataset) & Training the Model"""

from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Softmax, Flatten, Dense, BatchNormalization 
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras import regularizers
#import tensorflow as tf
from keras.models import Sequential

from keras import layers

#from keras.layers import Input, Dense
from keras import layers
from keras import optimizers
from keras.applications.densenet import DenseNet121

conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, channels), classes=2)

conv_base.summary()

from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

#from keras.callbacks import TensorBoard, Callback, EarlyStopping

#callbacks_list = [EarlyStopping(monitor='val_acc', patience=8, verbose=1)]

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)#,
      #callbacks=callbacks_list+[MetricsCheckpoint('logs')])

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

filenames = test_generator.filenames
print(filenames)
nb_samples = len(filenames)

percentCorrect = model.evaluate_generator(test_generator, steps = np.ceil(nb_samples / batch_size))
print(percentCorrect)

model.save('/hdd/kavi/test/denseNet121TranferLearnDataAugLab.h5')
