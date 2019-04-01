import numpy as np
import matplotlib.pyplot as plt
import csv

def readResultFile(fileLocation):

    experiment_number = []
    day_image_status = []
    precision = []
    recall = []
    fscore = []
    error = []

    with open(fileLocation) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader) # skip the first line
        for row in csvReader:
            experiment_number.append(float(row[0]))
            day_image_status.append(float(row[1]))
            precision.append(float(row[2]))
            recall.append(float(row[3]))
            fscore.append(float(row[4]))
            error.append(float(row[5]))

    experiment_number = np.array(experiment_number)
    day_image_status = np.array(day_image_status)
    precision = np.array(precision)
    recall = np.array(recall)
    fscore = np.array(fscore)
    error = np.array(error)

    return experiment_number, day_image_status, precision, recall, fscore, error



file_location = './results/balanced_experiments/result.txt'

(experiment_number, day_image_status, precision, recall, fscore, error) = readResultFile(file_location)
print (fscore)


day_index = np.where(day_image_status==1)
night_index = np.where(day_image_status==0)


# Combined statistics
print ('For combined Day + Night')
print ('precision = %s with +- %s' %(np.median(precision),   np.percentile(precision, 97.5) - np.percentile(precision, 2.5)))
print ('recall = %s with +- %s' %(np.median(recall),   np.percentile(recall, 97.5) - np.percentile(recall, 2.5)))
print ('fscore = %s with +- %s' %(np.median(fscore),   np.percentile(fscore, 97.5) - np.percentile(fscore, 2.5)))
print ('error = %s with +- %s' %(np.median(error),   np.percentile(error, 97.5) - np.percentile(error, 2.5)))



# Day values
exp_day = experiment_number[day_index]
prec_day = precision[day_index]
rec_day = recall[day_index]
fs_day = fscore[day_index]
err_day = error[day_index]

print ('For Day')
print ('precision = %s with +- %s' %(np.median(prec_day),   np.percentile(prec_day, 97.5) - np.percentile(prec_day, 2.5)))
print ('recall = %s with +- %s' %(np.median(rec_day),   np.percentile(rec_day, 97.5) - np.percentile(rec_day, 2.5)))
print ('fscore = %s with +- %s' %(np.median(fs_day),   np.percentile(fs_day, 97.5) - np.percentile(fs_day, 2.5)))
print ('error = %s with +- %s' %(np.median(err_day),   np.percentile(err_day, 97.5) - np.percentile(err_day, 2.5)))



# Night values
exp_night = experiment_number[night_index]
prec_night = precision[night_index]
rec_night = recall[night_index]
fs_night = fscore[night_index]
err_night = error[night_index]

print ('For Night')
print ('precision = %s with +- %s' %(np.median(prec_night),   np.percentile(prec_night, 97.5) - np.percentile(prec_night, 2.5)))
print ('recall = %s with +- %s' %(np.median(rec_night),   np.percentile(rec_night, 97.5) - np.percentile(rec_night, 2.5)))
print ('fscore = %s with +- %s' %(np.median(fs_night),   np.percentile(fs_night, 97.5) - np.percentile(fs_night, 2.5)))
print ('error = %s with +- %s' %(np.median(err_night),   np.percentile(err_night, 97.5) - np.percentile(err_night, 2.5)))





