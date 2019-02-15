import pandas as pd
import matplotlib as plt
import numpy as np

def normalize1D(array):
    #takes 1-d array of input data and normalizes it
    max_num = np.amax(array)
    array = array/max_num;
    return(array)

def normalizeAll(array):
    # takes array and normalizes each column separately
    for i in range(len(array[0, :])):
        array[:, i] = normalize1D(array[:, i])
    return array

def addNoise(array, error):
    # takes array of input data that needs noise added
    # error is the max ± error added to the dataset
    # as percetange of maximum value of row/col
    rows = len(array[:, 0])
    cols = len(array[0, :])

    # adds error to each column
    for j in range(cols):
        max_num = np.amax(array[:, j])
        delta = 2 * error * max_num * (np.random.rand(rows) - 0.5)
        array[:, j] = array[:, j] + delta
    return (array)


#import data
filename = "GlassData.csv"
rawData = np.genfromtxt(filename, delimiter = ',', skip_header = 1)
rawData = rawData[:,1:len(rawData[0,:])] #trim first col of data
rows = len(rawData[:,0])
cols = len(rawData[0,:])

#randomize data
newSet = rawData.copy()
np.random.shuffle(newSet)
newSet = normalizeAll(newSet)


#separate data into inputs and answers
answers = newSet[:,cols-1]
temp = np.ones(rows)
inputs = newSet.copy()
inputs[:,-1] = temp     #add bias

#add noise to data set
noise = 0.05    #noise is 5% of max value
inputs = addNoise(inputs,noise)

#separate data into training, validation and test sets
train_set = inputs[:150,:]  #length 150 (70%)
valid_set = inputs[150:182,:] #length 32 (15%)
test_set = inputs[182:214,:] #length 32 (15%)

np.savetxt("./training.csv", train_set, delimiter=",")
np.savetxt("./test.csv", test_set, delimiter=",")
np.savetxt("./valid.csv", valid_set, delimiter=",")
