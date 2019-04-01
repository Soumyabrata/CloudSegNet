# Import the libraries

from __future__ import division
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import h5py
import numpy as np
from scipy.misc import imread, imresize
import os
import os.path as op
import cv2
import scipy


# Resizing the ground-truth images into tensor vector
NO_OF_IMAGES = 115

batches = []

NO_OF_AUGS = 5

for i in range(NO_OF_IMAGES):


    if len(str(i+1))==1:
        image_name = '00'+str(i+1)+'_GT.jpg'
    elif len(str(i+1))==2:
        image_name = '0'+str(i+1)+'_GT.jpg'
    else:
        image_name = str(i+1)+'_GT.jpg'

    image_location = './dataset/SWINSEG/GTmaps/' + image_name
    print (['Reading image ',image_location])

    scene_image = cv2.imread(image_location,0)
    resized_image = scipy.misc.imresize(scene_image, (300, 300), interp='nearest').astype('float32')

    resized_image[resized_image < 128] = 0
    resized_image[resized_image == 128] = 0
    resized_image[resized_image > 128] = 255

    resized_image = np.expand_dims(resized_image, axis=2)

    # appending all images
    batches.append(resized_image)


    # Now adding augmented images too.
    for j in range(NO_OF_AUGS):
        aug_img_name = image_name[:-7] + '_' + str(j) + '.jpg'
        aug_img_location = './dataset/aug_SWINSEG/GTmaps/' + aug_img_name

        print (['Reading augmented image ', aug_img_location])

        scene_image = cv2.imread(aug_img_location, 0)
        resized_image = scipy.misc.imresize(scene_image, (300, 300), interp='nearest').astype('float32')
        resized_image[resized_image < 128] = 0
        resized_image[resized_image == 128] = 0
        resized_image[resized_image > 128] = 255
        resized_image = np.expand_dims(resized_image, axis=2)

        # appending all images
        batches.append(resized_image)

batches = np.array(batches, dtype=np.uint8)
print (batches.shape)


# saving the file
h5f = h5py.File('./data/night_images/night_withAUG_GT.h5', 'w')
h5f.create_dataset('GTmasks', data=batches)
h5f.close()


h5f = h5py.File('./data/night_images/night_withAUG_GT.h5','r')
GTmasks = h5f['GTmasks'][:]
h5f.close()
print (GTmasks.shape)




# Resizing the input images into tensor vector

NO_OF_IMAGES = 115

batches = []

for i in range(NO_OF_IMAGES):


    if len(str(i+1))==1:
        image_name = '00'+str(i+1)+'.jpg'
    elif len(str(i+1))==2:
        image_name = '0'+str(i+1)+'.jpg'
    else:
        image_name = str(i+1)+'.jpg'

    image_location = './dataset/SWINSEG/images/' + image_name
    print (['Reading scene image ',image_location])

    scene_image = imread(image_location)
    resized_image = scipy.misc.imresize(scene_image, (300, 300), interp='nearest').astype('float32')



    # appending all images
    batches.append(resized_image)

    # Now adding augmented images too.
    for j in range(NO_OF_AUGS):
        aug_img_name = image_name[:-4] + '_' + str(j) + '.jpg'
        aug_img_location = './dataset/aug_SWINSEG/images/' + aug_img_name

        print (['Reading augmented image ', aug_img_location])

        scene_image = imread(aug_img_location)
        resized_image = scipy.misc.imresize(scene_image, (300, 300), interp='nearest').astype('float32')
        print (resized_image.shape)

        # appending all images
        batches.append(resized_image)

batches = np.array(batches, dtype=np.uint8)
print (batches.shape)


# saving the file
h5f = h5py.File('./data/night_images/night_scene_withAUG.h5', 'w')
h5f.create_dataset('sceneimage', data=batches)
h5f.close()
print ('HDF file saved')


h5f = h5py.File('./data/night_images/night_scene_withAUG.h5', 'r')
sceneimage = h5f['sceneimage'][:]
h5f.close()
print (sceneimage.shape)