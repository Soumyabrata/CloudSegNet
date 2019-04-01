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
NO_OF_IMAGES = 1013
batches = []

NO_OF_AUGS = 5

for i in range(NO_OF_IMAGES):

    if len(str(i+1))==1:
        image_name = '000'+str(i+1)+'_GT.png'
    elif len(str(i+1))==2:
        image_name = '00'+str(i+1)+'_GT.png'
    elif len(str(i+1))==3:
        image_name = '0'+str(i+1)+'_GT.png'
    else:
        image_name = str(i+1)+'_GT.png'

    image_location = './dataset/SWIMSEG/GTmaps/' + image_name
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
        aug_img_location = './dataset/aug_SWIMSEG/GTmaps/' + aug_img_name

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
h5f = h5py.File('./data/day_images/day_withAUG_GT.h5', 'w')
h5f.create_dataset('GTmasks', data=batches)
h5f.close()


h5f = h5py.File('./data/day_images/day_withAUG_GT.h5','r')
GTmasks = h5f['GTmasks'][:]
h5f.close()
print (GTmasks.shape)






# Resizing the input images into tensor vector
NO_OF_IMAGES = 1013

batches = []

for i in range(NO_OF_IMAGES):

    if len(str(i+1))==1:
        image_name = '000'+str(i+1)+'.png'
    elif len(str(i+1))==2:
        image_name = '00'+str(i+1)+'.png'
    elif len(str(i+1))==3:
        image_name = '0'+str(i+1)+'.png'
    else:
        image_name = str(i+1)+'.png'


    image_location = './dataset/SWIMSEG/images/' + image_name
    print (['Reading scene image ',image_location])

    scene_image = imread(image_location)
    resized_image = scipy.misc.imresize(scene_image, (300, 300), interp='nearest').astype('float32')


    # appending all images
    batches.append(resized_image)


    # Now adding augmented images too.
    for j in range(NO_OF_AUGS):
        aug_img_name = image_name[:-4] + '_' + str(j) + '.jpg'
        aug_img_location = './dataset/aug_SWIMSEG/images/' + aug_img_name

        print (['Reading augmented image ', aug_img_location])

        scene_image = imread(aug_img_location)
        resized_image = scipy.misc.imresize(scene_image, (300, 300), interp='nearest').astype('float32')

        # appending all images
        batches.append(resized_image)


batches = np.array(batches, dtype=np.uint8)
print (batches.shape)


# saving the file
h5f = h5py.File('./data/day_images/day_scene_withAUG.h5', 'w')
h5f.create_dataset('sceneimage', data=batches)
h5f.close()
print ('HDF file saved')


h5f = h5py.File('./data/day_images/day_scene_withAUG.h5', 'r')
sceneimage = h5f['sceneimage'][:]
h5f.close()
print (sceneimage.shape)