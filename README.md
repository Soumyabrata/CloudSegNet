# CloudSegNet: A Deep Network for Nychthemeron Cloud Segmentation

With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> S. Dev, A. Nautiyal, Y. H. Lee, S. Winkler, CloudSegNet: A Deep Network for Nychthemeron Cloud Segmentation, IEEE Geoscience and Remote Sensing Letters, 2019.

![summary](./results/adsummary.png)

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.


## Usage

1. Create the following folder structure inside `./dataset`. 
    + `./dataset/SWIMSEG`: Contains the SWIMSEG dataset. The corresponding images, along with the corresponding ground-truth maps can be downloaded from [this](http://vintage.winklerbros.net/swimseg.html) link. The images are saved inside `./dataset/SWIMSEG/images` folder, and the corresponding ground-truth maps are saved inside `./dataset/SWIMSEG/GTmaps`.
    + `./dataset/SWINSEG`: Contains the SWINSEG dataset. The dataset can be downloaded from [this](http://vintage.winklerbros.net/swinseg.html) link. The images and ground-truth maps are saved in the same order.
    + `./dataset/aug_SWIMSEG`: Contains the augmented set of daytime images. It follows the similar structure, and is computed using the script `create_aug_day.py`.
    + `./dataset/aug_SWINSEG`: Contains the augmented set of nightime images. It follows the similar structure, and is computed using the script `create_aug_night.py`. 
2. Run the script `python2 create_aug_day.py`. Please install `keras 2.0.0` using `pip install keras==2.0.0`. The latest version of keras has `image` module removed, and re-factored into `ImageDataGenerator` class, and therefore, renders the current version error-prone (more details [here](https://stackoverflow.com/questions/51311062/cant-import-apply-transform-from-keras-preprocessing-image)). You have to change this script, if you wish to make it compatible with the latest version of keras. It saves the augmented images inside `aug_SWIMSEG/images/` and corresponding augmented masks inside `aug_SWIMSEG/GTmaps/`.  
3. Run the script `python2 create_aug_night.py`, for computing the augmentated images and masks for nighttime images. This saves the augmented images inside `aug_SWINSEG/images/` folder, and corresponding masks in `aug_SWINSEG/GTmaps/`. 
4. Run the script `python2 create_dayimages.py` for generating the `.h5` file for actual- and augmented- daytime images. The results are stored in `./data/day_images`.
5. Run the script `python2 create_nightimages.py` for generating the `.h5` file for actual- and augmented- nightime images. The results are stored in `./data/night_images`.
6. Run the script `python2 train_model.py` for training the CloudSegNet model in the composite dataset containing actual- and augmented- images. The logfile and the model is saved inside the folder `./results/withAUG_dataset`.
7. Run the script `python2 train_model_balanced.py` for training the CloudSegNet model in a balanced dataset with equal number of day- and night- images. All night images of extended dataset are considered, and a single random sample of day images included for the training.composite dataset containing actual- and augmented- images. The logfile and the model is saved inside the folder `./results/balanced_random_sample`.
8. Run the script `python2 evaluation_DNN.py` for getting the evaluation results of CloudSegNet model. The values are populated in Table I.
9. Run the script `python2 generate_figures.py` for generating Figure 2 and Figure 3 of the paper. 
10. Run the script `python2 result_rand_exps.py` for getting results of Table II. The results are stored in `./results/balanced_experiments` folder. Subsequently, run the script `python2 sensitivity.py` to compute the Table II values.


### Additional helper scripts

+ calculate_score.py: Helper script needed during ROC computation
+ create_dataset.py: Splits datasets into training and testing sets
+ roc_items.py: Helper script during ROC computation
+ score_card.py: Computes the different evaluation scores