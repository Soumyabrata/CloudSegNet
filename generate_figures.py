import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './scripts/')

import matplotlib.cm as cm
from calculate_score import *
from score_card import *

f = open('./results/balanced_random_sample/logfile.txt', 'r')
x = f.readlines()

epoch_array = []
training_array = []
testing_array = []

for line in x[1:]:
    items = line.split(',')
    epoch_array.append(float(items[0]))
    training_array.append(float(items[1]))
    testing_array.append(float(items[2]))

epoch_array = np.array(epoch_array)
training_array = np.array(training_array)
testing_array = np.array(testing_array)

plt.figure(1, figsize=(6,4))
plt.plot(np.log(training_array),'b',label='Training set')
plt.plot(np.log(testing_array),'r',label='Testing Set', alpha=0.4)
plt.xlabel('Number of epochs', fontsize=14)
plt.ylabel('Loss [in log scale]', fontsize=14)
plt.grid(True)
#plt.ylim([-1.75,0.5])
plt.legend(loc='upper right')
plt.savefig("./results/loss-plot.pdf", bbox_inches='tight')




X_testing = np.load('./results/balanced_random_sample/xtesting.npy')
print ('from the saved data')
print (X_testing.shape)

Y_testing = np.load('./results/balanced_random_sample/ytesting.npy')
imagetypetesting = np.load('./results/balanced_random_sample/imagetypetesting.npy')

from roc_items import *
from keras.models import load_model

ae = load_model('./results/balanced_random_sample/cloudsegnet.hdf5')

roc_values = np.arange(0.0, 1.01, 0.01)

TPR_values = np.zeros(len(roc_values))
FPR_values = np.zeros(len(roc_values))

marker_values = np.zeros([11, 2])
marker_index = 0

(no_of_testing_images, _, _, _) = X_testing.shape

precision_array = []
recall_array = []
fscore_array = []
error_array = []

pop_counter = 11
for i, threshold in enumerate(roc_values):

    tpr_each_image = np.zeros(no_of_testing_images)
    fpr_each_image = np.zeros(no_of_testing_images)

    print (['checking for threshold ', str(threshold)])
    pop_counter = pop_counter - 1

    for sample_iter in range(no_of_testing_images):
        gt_image = Y_testing[sample_iter]
        gt_image = np.squeeze(gt_image)
        input_image = X_testing[sample_iter]
        image_map = calculate_map(input_image, ae)
        (tpr, fpr) = calculate_score_threshold(image_map, gt_image, threshold)
        tpr_each_image[sample_iter] = tpr
        fpr_each_image[sample_iter] = fpr

    # averaging
    tpr_each_image = np.array(tpr_each_image)
    fpr_each_image = np.array(fpr_each_image)

    TPR_values[i] = np.mean(tpr_each_image)
    FPR_values[i] = np.mean(fpr_each_image)

    if (threshold == 0) or (pop_counter == 0):
        print (['populating for ', str(threshold)])
        marker_values[marker_index, 0] = np.mean(fpr_each_image)
        marker_values[marker_index, 1] = np.mean(tpr_each_image)
        print (marker_values)
        marker_index = marker_index + 1
        pop_counter = 10

    # Saving the ROC values
np.save('./results/balanced_random_sample/TPR_values', TPR_values)
np.save('./results/balanced_random_sample/FPR_values', FPR_values)
np.save('./results/balanced_random_sample/marker_values', marker_values)

data = np.load('./results/balanced_random_sample/TPR_values.npy')
print ('from saved data')
print (data.shape)


TPR_values = np.load('./results/balanced_random_sample/TPR_values.npy')
FPR_values = np.load('./results/balanced_random_sample/FPR_values.npy')
marker_values = np.load('./results/balanced_random_sample/marker_values.npy')

print (marker_values)

# Plotting the ROC figure
plt.figure(2,figsize=(4,4))
plt.scatter(marker_values[:,0], marker_values[:,1])
plt.plot(FPR_values,TPR_values, 'b')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.grid(True)
plt.savefig("./results/roc.pdf", bbox_inches='tight')


# Area under ROC curve
from sklearn import metrics
print (metrics.auc(FPR_values, TPR_values))