import numpy as np

def calculate_map(input_image, saved_model):


    test_image = np.expand_dims(input_image, axis=0)
    show_test_image = np.squeeze(test_image)

    decoded_img = saved_model.predict(test_image)
    show_decoded_image = np.squeeze(decoded_img)


    return show_decoded_image


def calculate_score_threshold(input_map, groundtruth_image, threshold):


    binary_map = input_map
    binary_map[binary_map < threshold ] = 0
    binary_map[binary_map == threshold ] = 0
    binary_map[binary_map > threshold ] = 1

    [rows,cols] = groundtruth_image.shape
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i,j]==1 and binary_map[i,j]==1): #TP condition
                TP = TP + 1
            elif ((groundtruth_image[i,j]==0) and (binary_map[i,j]==1)): #FP condition
                FP = FP + 1
            elif ((groundtruth_image[i,j]==0) and (binary_map[i,j]==0)): #TN condition
                TN = TN + 1
            elif ((groundtruth_image[i,j]==1) and (binary_map[i,j]==0)): #FN condition
                FN = FN + 1


    tpr = float(TP)/float(TP+FN)
    fpr = float(FP)/float(FP+TN)

    return (tpr, fpr)