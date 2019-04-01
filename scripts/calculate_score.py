import numpy as np
from sklearn.cluster import KMeans

def calculate_score(input_image, groundtruth_image, saved_model):


    test_image = np.expand_dims(input_image, axis=0)
    show_test_image = np.squeeze(test_image)

    decoded_img = saved_model.predict(test_image)
    show_decoded_image = np.squeeze(decoded_img)
    #print (decoded_img.shape)



    data = show_decoded_image
    flat_data = data.reshape(-1,1)
    print (flat_data.shape)

    kmeans = KMeans(n_clusters=2)
    # Fitting the input data
    kmeans = kmeans.fit(flat_data)
    # Getting the cluster labels
    labels = kmeans.predict(flat_data)
    
    centroids = kmeans.cluster_centers_

    cen1 = centroids[0][0]
    cen2 = centroids[1][0]

    #print (cen1,cen2)


    label_mat = labels.reshape(300,300)

    if cen1 > cen2:
        label_mat[label_mat == 0] = 5
        label_mat[label_mat == 1] = 0
        label_mat[label_mat == 5] = 1




    # Calculation of the scores
    #groundtruth_image[groundtruth_image < 128 ] = 0
    #groundtruth_image[groundtruth_image == 128 ] = 0
    #groundtruth_image[groundtruth_image > 128 ] = 1

    [rows,cols] = groundtruth_image.shape
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i,j]==1 and label_mat[i,j]==1): #TP condition
                TP = TP + 1
            elif ((groundtruth_image[i,j]==0) and (label_mat[i,j]==1)): #FP condition
                FP = FP + 1
            elif ((groundtruth_image[i,j]==0) and (label_mat[i,j]==0)): #TN condition
                TN = TN + 1
            elif ((groundtruth_image[i,j]==1) and (label_mat[i,j]==0)): #FN condition
                FN = FN + 1

    
    #print (TP,FP,TN,FN)

    precision=float(TP)/float(TP+FP)
    #print (precision)

    recall=float(TP)/float(TP+FN)
    #print (recall)

    fScore=float(2*precision*recall)/float(precision+recall)


    error_count = 0
    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i,j] != label_mat[i,j]):
                error_count = error_count + 1

    error_rate = float(error_count)/float(rows*cols)


    return (label_mat, precision, recall, fScore, error_rate)