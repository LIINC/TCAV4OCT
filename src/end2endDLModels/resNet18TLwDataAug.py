"""# Loading Train, Val, and Test Data Using Keras ImageDataGenerator"""

# Directory Paths
#train_dir = "/gdrive/My Drive/newCircleData/Train/"
#val_dir = "/gdrive/My Drive/newCircleData/Val/"
test_dir = "/hdd/kavi/test/mapsRedCap_DCH/fullReports/" #"/gdrive/My Drive/mapsRedCap_DCH/fullReports/" 
train_data_dir = "/hdd/kavi/test/newCircleData/TrainValTest/"

import numpy as np 
import tensorflow as tf
#from numpy.random import seed
#tf.random.set_seed(seed(2))

#Image dims and training details
img_width = 600
img_height = 450
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

# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)

# train_generator = train_datagen.flow_from_directory(
#     train_dir, 
#     target_size=(img_height, img_width),
#     color_mode="rgb",
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=True)   

# valid_generator = valid_datagen.flow_from_directory(
#     val_dir,
#     target_size=(img_height, img_width),
#     color_mode="rgb",
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=True) #weight toward one class or another

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

"""# Building the Model Architecture (Pre-trained ResNet-18 Extracting Features from OCT Dataset) & Training the Model"""

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

from keras.callbacks import TensorBoard

#from keras.layers import Input, Dense
from keras import layers
from keras import optimizers
#from keras.applications.inception_v3 import InceptionV3
from classification_models import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')

#from classfication_models import classfication_models
#from classification_models.classification_models.resnet import models
#from classification_models import models

# for keras
#from classification_models.keras import Classifiers
# for tensorflow.keras
conv_base = ResNet18(weights='imagenet', include_top=False, input_shape=(img_height, img_width, channels), classes=2)

conv_base.summary()

from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.add(Conv2D(32,(11, 11), input_shape=input_shape))#, kernel_regularizer=regularizers.l1(0.01))) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

# model.add(Conv2D(32,(3, 3), input_shape=input_shape))#, kernel_regularizer=regularizers.l1(0.01))) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

# model.add(Conv2D(32,(3, 3), input_shape=input_shape))#, kernel_regularizer=regularizers.l1(0.01))) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

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

model.save('/hdd/kavi/test/resNet18TranferLearnDataAug063020.h5')
