# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:24:30 2017

@author: Admin
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle

# train model
# Divide up into cars and notcars
    
# Read Cars Images
images = glob.glob('../vehicles/*/*.png')
cars = []
y = []
for imagen in images:
    image = cv2.imread(imagen)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cars.append(image)
    y.append(1)
#images = glob.glob('../vehicles/GTI_Left/*.png')
#for image in images:
#   cars.append(image)

    
# Read Not Cars Images
images = glob.glob('../non-vehicles/*/*.png')
notcars = []
for imagen in images:
    image = cv2.imread(imagen)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    notcars.append(image)
    y.append(0)



X_train = np.vstack((cars, notcars)).astype(np.float64)                        

y_train = y
# Define the labels vector
#y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
# import BatchNormalization
from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D

model = Sequential()
# data normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape =(64,64,3)))

# cropping to take only the relevant parts of the images
#model.add(Cropping2D(cropping=((40,20), (0,0))))

""" NN Model according to NVIDIA: 
5 Convolution layers and 4 fully connected layers with dropping as regularization""" 
DRATE = 0.2 # dropp out rate

model.add(Convolution2D(24,5,5, subsample = (2,2), activation="relu"))
model.add(Dropout(DRATE))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation="relu"))
model.add(Dropout(DRATE))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation="relu"))
model.add(Dropout(DRATE))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(DRATE))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(DRATE))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(DRATE))    
model.add(Dense(50))
model.add(Dropout(DRATE))
model.add(Dense(10))
model.add(Dropout(DRATE))
model.add(Dense(1))

# use mean square error cost function and adam optimizer 
model.compile(loss='mse', optimizer = 'adam')

from keras.callbacks import ModelCheckpoint, EarlyStopping

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(filepath='weights.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
]

# train model
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch = 20,
          callbacks=callbacks)
# save model
model.save('model.h5')



from keras.models import load_model
#import h5py
model = load_model('model.h5')
image_array = np.asarray(X_train[1])
pred =  float(model.predict(image_array[None, :, :, :], batch_size=1))
print(pred)  

 