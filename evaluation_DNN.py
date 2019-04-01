# This provides the results of Table I in the paper.

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.misc
from keras.models import load_model

import sys
sys.path.insert(0, './scripts/')

from score_card import *
from roc_items import *

# This discusses the results of combined dataset
#RESULT_FOLDER = './results/combined/'

# This discusses the results of balanced ramdom sample
RESULT_FOLDER = './results/balanced_random_sample/'


ae = load_model(RESULT_FOLDER + 'cloudsegnet.hdf5')



X_testing = np.load(RESULT_FOLDER + 'xtesting.npy')
print ('from the saved data')
print (X_testing.shape)


Y_testing = np.load(RESULT_FOLDER + 'ytesting.npy')
imagetypetesting = np.load(RESULT_FOLDER + 'imagetypetesting.npy')



(no_of_testing_images, _, _, _) = X_testing.shape

for sample_iter in range(no_of_testing_images):

    if imagetypetesting[sample_iter] == 1: #day image
        image_array = X_testing[sample_iter]
        gt_array = Y_testing[sample_iter]
        gt_array = np.squeeze(gt_array)



        save_img_name = RESULT_FOLDER + 'testing_images/day/img/img' + str(sample_iter) + '.jpg'
        scipy.misc.imsave(save_img_name, image_array)

        save_gt_name = RESULT_FOLDER + 'testing_images/day/GT/img' + str(sample_iter) + '_GT.png'
        plt.imsave(save_gt_name, gt_array, cmap=cm.gray)


    elif imagetypetesting[sample_iter] == 0: #night image
        image_array = X_testing[sample_iter]
        gt_array = Y_testing[sample_iter]
        gt_array = np.squeeze(gt_array)

        save_img_name = RESULT_FOLDER + 'testing_images/night/img/img' + str(sample_iter) + '.jpg'
        scipy.misc.imsave(save_img_name, image_array)

        save_gt_name = RESULT_FOLDER + 'testing_images/night/GT/img' + str(sample_iter) + '_GT.png'
        plt.imsave(save_gt_name, gt_array, cmap=cm.gray)






## Evaluate - all time images
(no_of_testing_images, _, _, _) = X_testing.shape

precision_array = []
recall_array = []
fscore_array = []
error_array = []

threshold_value = 0.5

for sample_iter in range(no_of_testing_images):
    gt_image = Y_testing[sample_iter]
    gt_image = np.squeeze(gt_image)

    input_image = X_testing[sample_iter]
    image_map = calculate_map(input_image, ae)


    (precision, recall, fScore, error_rate) = score_card(image_map, gt_image, threshold_value)

    precision_array.append(precision)
    recall_array.append(recall)
    fscore_array.append(fScore)
    error_array.append(error_rate)

    print (sample_iter, fScore, error_rate)


precision_array = np.array(precision_array)
recall_array = np.array(recall_array)
fscore_array = np.array(fscore_array)
error_array = np.array(error_array)

print (['Alltime precision = ', str(np.mean(precision_array))])
print (['Alltime recall = ', str(np.mean(recall_array))])
print (['Alltime fscore = ', str(np.mean(fscore_array))])
print (['Alltime error = ', str(np.mean(error_array))])






## Evaluate - only NIGHT images

(no_of_testing_images, _, _, _) = X_testing.shape

precision_array = []
recall_array = []
fscore_array = []
error_array = []

threshold_value = 0.5

for sample_iter in range(no_of_testing_images):

    if imagetypetesting[sample_iter] == 0: #night image

        gt_image = Y_testing[sample_iter]
        gt_image = np.squeeze(gt_image)

        input_image = X_testing[sample_iter]
        image_map = calculate_map(input_image, ae)


        (precision, recall, fScore, error_rate) = score_card(image_map, gt_image, threshold_value)

        precision_array.append(precision)
        recall_array.append(recall)
        fscore_array.append(fScore)
        error_array.append(error_rate)

        print (sample_iter, fScore, error_rate)


precision_array = np.array(precision_array)
recall_array = np.array(recall_array)
fscore_array = np.array(fscore_array)
error_array = np.array(error_array)

print (['Night precision = ', str(np.mean(precision_array))])
print (['Night recall = ', str(np.mean(recall_array))])
print (['Night fscore = ', str(np.mean(fscore_array))])
print (['Night error = ', str(np.mean(error_array))])





## Evaluate - only DAY images

(no_of_testing_images, _, _, _) = X_testing.shape

precision_array = []
recall_array = []
fscore_array = []
error_array = []

threshold_value = 0.5

for sample_iter in range(no_of_testing_images):

    if imagetypetesting[sample_iter] == 1: #day image

        gt_image = Y_testing[sample_iter]
        gt_image = np.squeeze(gt_image)

        input_image = X_testing[sample_iter]
        image_map = calculate_map(input_image, ae)


        (precision, recall, fScore, error_rate) = score_card(image_map, gt_image, threshold_value)

        precision_array.append(precision)
        recall_array.append(recall)
        fscore_array.append(fScore)
        error_array.append(error_rate)

        print (sample_iter, fScore, error_rate)


precision_array = np.array(precision_array)
recall_array = np.array(recall_array)
fscore_array = np.array(fscore_array)
error_array = np.array(error_array)

print (['Day precision = ', str(np.mean(precision_array))])
print (['Day recall = ', str(np.mean(recall_array))])
print (['Day fscore = ', str(np.mean(fscore_array))])
print (['Day error = ', str(np.mean(error_array))])