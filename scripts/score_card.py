import numpy as np

def score_card(input_map, groundtruth_image, threshold):


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



    precision=float(TP)/float(TP+FP)
    recall=float(TP)/float(TP+FN)
    fScore=float(2*precision*recall)/float(precision+recall)


    error_count = 0
    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i,j] != binary_map[i,j]):
                error_count = error_count + 1
    error_rate = float(error_count)/float(rows*cols)



    return (precision, recall, fScore, error_rate)