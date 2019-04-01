import numpy as np


def randomize_data(original_img_hdf, original_mask_hdf,  percentage_training, percentage_testing):

    (number_of_original, _, _, _) = original_img_hdf.shape

    number_of_training = (percentage_training/100.0)*number_of_original
    number_of_testing = (percentage_testing / 100.0) * number_of_original

    number_of_training = int(number_of_training)
    number_of_testing = int(number_of_testing)

    print ('Number of training Parent images = ',number_of_training)
    print ('Number of testing Parent images = ', number_of_testing)

    a = np.arange(number_of_original)
    np.random.shuffle(a)


    index_of_training = a[:number_of_training]
    index_of_testing = a[number_of_training: number_of_training+number_of_testing]

    X_train = original_img_hdf[index_of_training]
    Y_train = original_mask_hdf[index_of_training]

    X_testing = original_img_hdf[index_of_testing]
    Y_testing = original_mask_hdf[index_of_testing]

    return (X_train, X_testing, Y_train, Y_testing)





def randomize_data_ratio(original_img_hdf, original_mask_hdf,  percentage_training, percentage_testing):

    (number_of_original, _, _) = original_img_hdf.shape

    number_of_training = (percentage_training/100.0)*number_of_original
    number_of_testing = (percentage_testing / 100.0) * number_of_original

    number_of_training = int(number_of_training)
    number_of_testing = int(number_of_testing)

    print ('Number of training Parent images = ',number_of_training)
    print ('Number of testing Parent images = ', number_of_testing)

    a = np.arange(number_of_original)
    np.random.shuffle(a)


    index_of_training = a[:number_of_training]
    index_of_testing = a[number_of_training: number_of_training+number_of_testing]


    X_train = original_img_hdf[index_of_training]
    Y_train = original_mask_hdf[index_of_training]

    X_testing = original_img_hdf[index_of_testing]
    Y_testing = original_mask_hdf[index_of_testing]

    return (X_train, X_testing, Y_train, Y_testing)







def randomize_data_alltimes(original_img_hdf, original_mask_hdf,  no_of_dayimages, no_of_nightimages, percentage_training, percentage_testing):

    #no_of_dayimages = 1013
    #no_of_nightimages = 115

    # Day = 1 and night = 0 are the labels
    a = np.ones(no_of_dayimages)
    b = np.zeros(no_of_nightimages)

    image_type_array =  np.concatenate([a,b])
    print (image_type_array.shape)

    (number_of_original, _, _, _) = original_img_hdf.shape

    number_of_training = (percentage_training/100.0)*number_of_original
    number_of_testing = (percentage_testing / 100.0) * number_of_original

    number_of_training = int(number_of_training)
    number_of_testing = int(number_of_testing)

    print ('Number of training Parent images = ',number_of_training)
    print ('Number of testing Parent images = ', number_of_testing)

    a = np.arange(number_of_original)
    np.random.shuffle(a)
    print (number_of_original)


    index_of_training = a[:number_of_training]
    index_of_testing = a[number_of_training: ]
    print (index_of_testing)

    X_train = original_img_hdf[index_of_training]
    Y_train = original_mask_hdf[index_of_training]

    X_testing = original_img_hdf[index_of_testing]
    Y_testing = original_mask_hdf[index_of_testing]

    imagetype_testing = image_type_array[index_of_testing]


    return (X_train, X_testing, Y_train, Y_testing, imagetype_testing)

