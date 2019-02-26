"""
CMPE 452 Assignment 2
Backpropagating Neural Network
Curtis Shewchuk
SN: 10189026

Data processing functions used in the main. Separate file used when building for testing on a script (not included).
The main function that is used is the 'processData' function, which
"""
import numpy as np

def processData(data, numOutputs):
    """
    Randomize, normalize and process the data to create 3 data sets (train, test and valid) with equal
    distribution of data between the sets. Data is to be split:
    70%: Training
    15%: Test
    15%: Valid
    :param data: Input data to process (GlassData.csv)
    :param numOutputs: Number of required output nodes
    :return: 6 arrays of input data and expected output, properly processed by above description
    """

    # Sort the data input data by known output category and shuffle the data around
    np.random.shuffle(data[:70, :])
    np.random.shuffle(data[70:146, :])
    np.random.shuffle(data[146:163, :])
    np.random.shuffle(data[163:176, :])
    np.random.shuffle(data[176:185, :])
    np.random.shuffle(data[185:214, :])

    setOne = data[:70, :].copy()
    setTwo = data[70:146, :].copy()
    setThree = data[146:163, :].copy()
    setFour = data[163:176, :].copy()
    setFive = data[176:185, :].copy()
    setSix = data[185:214, :].copy()

    setOne = normalize2D(setOne)
    setTwo = normalize2D(setTwo)
    setThree = normalize2D(setThree)
    setFour = normalize2D(setFour)
    setFive = normalize2D(setFive)
    setSix = normalize2D(setSix)

    ## Now create the data set and their answer arrays

    trainSet = np.concatenate((setOne[0:49, :], setTwo[0:53, :], setThree[0:12, :], setFour[0:9, :], setFive[0:6, :], setSix[0:20, :]), axis=0)
    np.random.shuffle(trainSet)
    trainAns = answerVector(trainSet[:, -1], numOutputs, len(trainSet[:, 0]))
    trainSet[:, -1] = np.ones(len(trainSet[:, 0])) #Add the bias weight to the end column

    validSet = np.concatenate((setOne[49:59, :], setTwo[53:64, :], setThree[12:14, :], setFour[9:11, :], setFive[6:7, :], setSix[20:24, :]), axis=0)
    validAns = answerVector(validSet[:, -1], numOutputs, len(validSet[:, 0]))
    validSet[:, -1] = np.ones(len(validSet[:, 0]))

    testSet= np.concatenate((setOne[59:70, :], setTwo[64:76, :], setThree[14:17, :], setFour[11:13, :], setFive[7:9, :], setSix[24:29, :]), axis=0)
    testAns = answerVector(testSet[:, -1], numOutputs, len(testSet[:, 0]))
    testSet[:, -1] = np.ones(len(testSet[:, 0]))

    return trainSet, trainAns, validSet, validAns, testSet, testAns


def answerVector(answers, numOutputs, rows):
    """
    Generate an output vector of the form
    [0,0,0,0...numOutputs] where a 1 in the vector represents the expected output
    :param answer_set:
    :param n_out:
    :param rows:
    :return:
    """
    answerVector = np.zeros((rows,numOutputs))
    for i in range(len(answers)):
        if answers[i] == 1:
            answerVector[i,0] = 1
        elif answers[i] == 2:
            answerVector[i,1] = 1
        elif answers[i] == 3:
            answerVector[i,2] = 1
        elif answers[i] == 5:
            answerVector[i,3] = 1
        elif answers[i] == 6:
            answerVector[i,4] = 1
        else:
            answerVector[i,5] = 1

    return answerVector

def normalize(array):
    """
    Normalize a 1D array
    :param array: 1D Array to normalize
    :return: 1D Array. Normalized version of the input array
    """
    max_num = np.amax(array)
    if max_num == 0:
        array *= 0
    else:
        array = array / max_num
    return (array)


def normalize2D(array):
    """
    Normalizes a 2D array, ignoring the last column (assumed in the design that last column contains data we can't change at all)
    :param array: 2D array to normalize
    :return: 2D array, normalized
    """
    for i in range(len(array[0, :]) - 1):
        array[:, i] = normalize(array[:, i])
    return array

def addNoise(array, error):
    """
    Adds noise to an array based on the error percentage given
    :param array: Array to add noise to
    :param error: Error percentage to use
    :return: The array with noise added to the values in the array
    """
    rows = len(array[:, 0])
    cols = len(array[0, :])

    for i in range(cols - 1):
        maxNum = np.amax(array[:, i])
        sigma = 2 * error * maxNum * (np.random.rand(rows) - 0.5)
        array[:, i] = array[:, i] + sigma

    return (array)