# This provides results for Table II.

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



# All fixed parameters
NO_OF_EXPS = 10
NO_OF_EPOCHS = 400
threshold_value = 0.5



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

equal_number_of_images = no_of_nightimages






p_array = []
r_array = []
fs_array = []
e_array = []


day_p_array = []
day_r_array = []
day_fs_array = []
day_e_array = []


night_p_array = []
night_r_array = []
night_fs_array = []
night_e_array = []


text_file = open("./results/balanced_experiments/result.txt", "w")
text_file.write("experiment_number, day_image_status, precision, recall, fscore, error \n")



for ex in range(NO_OF_EXPS):


    print (['Performing experiment', str(ex+1), 'out of', str(NO_OF_EXPS)])

    # extracting a sample of day images to make a balanced dataset
    a = np.arange(no_of_nightimages)
    np.random.shuffle(a)

    uniform_day_img = original_sceneimage[a, :, :, :]
    uniform_day_gt = original_GTmasks[a, :, :, :]

    # Combining both day and night images in a single tensor
    scene_data = np.vstack([uniform_day_img, original_sceneimage_night])
    print (scene_data.shape)

    gt_data = np.vstack([uniform_day_gt, original_GTmasks_night])
    print (gt_data.shape)



    #===============================================================
    # Creating the dataset for training our model
    #===============================================================
    print ('Shuffling the dataset and creating the various sets')
    (X_train, X_testing, Y_train, Y_testing, imagetype_testing) = randomize_data_alltimes(scene_data, gt_data, equal_number_of_images, equal_number_of_images, percentage_training=80, percentage_testing=20)


    print (X_train.shape)
    print (X_testing.shape)


    print (Y_train.shape)
    print (Y_testing.shape)

    print (imagetype_testing)







    #===============================================================
    # Model training
    #===============================================================

    csv_logger = CSVLogger('./results/balanced_experiments/logfile.txt')
    '''
    saves the model weights after each epoch if the validation loss decreased
    '''
    checkpointer = ModelCheckpoint(filepath='./results/balanced_experiments/cloudsegnet.hdf5', verbose=1, save_best_only=True)
    autoencoder.fit(X_train, Y_train, epochs=NO_OF_EPOCHS, batch_size=32,
                    validation_data=(X_testing, Y_testing), verbose=1,callbacks=[csv_logger, checkpointer])






    # Test the results

    from score_card import *
    from roc_items import *



    (no_of_testing_images, _, _, _) = X_testing.shape

    precision_array = []
    recall_array = []
    fscore_array = []
    error_array = []

    day_precision_array = []
    day_recall_array = []
    day_fscore_array = []
    day_error_array = []

    night_precision_array = []
    night_recall_array = []
    night_fscore_array = []
    night_error_array = []



    from keras.models import load_model
    ae = load_model('./results/balanced_experiments/cloudsegnet.hdf5')

    for sample_iter in range(no_of_testing_images):

        # All time images

        gt_image = Y_testing[sample_iter]
        gt_image = np.squeeze(gt_image)

        input_image = X_testing[sample_iter]
        image_map = calculate_map(input_image, ae)


        (precision, recall, fScore, error_rate) = score_card(image_map, gt_image, threshold_value)

        precision_array.append(precision)
        recall_array.append(recall)
        fscore_array.append(fScore)
        error_array.append(error_rate)


        # Only day images
        if imagetype_testing[sample_iter] == 1:  # day image

            day_status = 1

            gt_image = Y_testing[sample_iter]
            gt_image = np.squeeze(gt_image)

            input_image = X_testing[sample_iter]
            image_map = calculate_map(input_image, ae)

            (precision, recall, fScore, error_rate) = score_card(image_map, gt_image, threshold_value)

            day_precision_array.append(precision)
            day_recall_array.append(recall)
            day_fscore_array.append(fScore)
            day_error_array.append(error_rate)

            text_file.write("%s, %s, %s, %s, %s, %s \n" % (ex, day_status, precision, recall, fScore, error_rate))


        # Only night images
        if imagetype_testing[sample_iter] == 0:  # night image

            day_status = 0

            gt_image = Y_testing[sample_iter]
            gt_image = np.squeeze(gt_image)

            input_image = X_testing[sample_iter]
            image_map = calculate_map(input_image, ae)

            (precision, recall, fScore, error_rate) = score_card(image_map, gt_image, threshold_value)

            night_precision_array.append(precision)
            night_recall_array.append(recall)
            night_fscore_array.append(fScore)
            night_error_array.append(error_rate)

            text_file.write("%s, %s, %s, %s, %s, %s \n" % (ex, day_status, precision, recall, fScore, error_rate))





text_file.close()