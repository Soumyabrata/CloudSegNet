import numpy as np




import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import train_test_split
import h5py
from keras.callbacks import ModelCheckpoint, CSVLogger


import sys
sys.path.insert(0, './scripts/')

from create_dataset import *



# ### Train the deep learning model
input_img = Input(shape=(300, 300, 3)) # 

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

print "shape of encoded", K.int_shape(encoded)



#==============================================================================


x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
x = UpSampling2D((2, 2))(x)

decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
print "shape of decoded", K.int_shape(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




#===============================================================
# Reading the DAY HDF files
#===============================================================

# original scene image
h5f = h5py.File('./data/day_images/day_scene_withAUG.h5','r')
original_sceneimage = h5f['sceneimage'][:]
h5f.close()
print "original scene hdf5 file's shape", original_sceneimage.shape

# original ground truth image
h5f = h5py.File('./data/day_images/day_withAUG_GT.h5','r')
original_GTmasks = h5f['GTmasks'][:]
h5f.close()
print "original ground truth hdf5 file's shape", original_GTmasks.shape
original_GTmasks = original_GTmasks.astype('float32')/255.



(no_of_dayimages, _, _, _) = original_sceneimage.shape
print (no_of_dayimages)

# -------------------------------------


# Reading the NIGHT HDF files

# original scene image
h5f = h5py.File('./data/night_images/night_scene_withAUG.h5','r')
original_sceneimage_night = h5f['sceneimage'][:]
h5f.close()
print "original scene hdf5 file's shape", original_sceneimage_night.shape


# original ground truth image
h5f = h5py.File('./data/night_images/night_withAUG_GT.h5','r')
original_GTmasks_night = h5f['GTmasks'][:]
h5f.close()
print "original ground truth hdf5 file's shape", original_GTmasks_night.shape
original_GTmasks_night = original_GTmasks_night.astype('float32')/255.




(no_of_nightimages, _, _, _) = original_sceneimage_night.shape
print (no_of_nightimages)

# Combining both day and night images in a single tensor
scene_data = np.vstack([original_sceneimage,original_sceneimage_night])
print (scene_data.shape)

gt_data = np.vstack([original_GTmasks,original_GTmasks_night])
print (gt_data.shape)




#===============================================================
# Creating the dataset for training our model
#===============================================================
print ('Shuffling the dataset and creating the various sets')
(X_train, X_testing, Y_train, Y_testing, imagetype_testing) = randomize_data_alltimes(scene_data, gt_data, no_of_dayimages, no_of_nightimages, percentage_training=80, percentage_testing=20)


print (X_train.shape)
print (X_testing.shape)


print (Y_train.shape)
print (Y_testing.shape)

print (imagetype_testing.shape)



# Saving the testing images and ground truths (as they are always randomized)
np.save('./results/withAUG_dataset/xtesting.npy', X_testing)
np.save('./results/withAUG_dataset/ytesting.npy', Y_testing)
np.save('./results/withAUG_dataset/imagetypetesting.npy', imagetype_testing)

data = np.load('./results/withAUG_dataset/xtesting.npy')
print ('from the saved data')
print (data.shape)



#===============================================================
# Model training
#===============================================================

csv_logger = CSVLogger('./results/withAUG_dataset/logfile.txt')
'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath='./results/withAUG_dataset/cloudsegnet.hdf5', verbose=1, save_best_only=True)
autoencoder.fit(X_train, Y_train, epochs=50000, batch_size=32,
                validation_data=(X_testing, Y_testing), verbose=1,callbacks=[csv_logger, checkpointer])


