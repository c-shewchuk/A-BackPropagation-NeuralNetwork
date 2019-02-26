"""
CMPE 452 Assignment 2
Backpropagating Neural Network
Curtis Shewchuk
SN: 10189026

The functions required for the back propagation in the network. Mostly supporting functions, which you will see
called in the main.py.
"""
import numpy as np


def trainNetwork(trainSet, trainAns, validSet, validAns):
    momentum = 1
    lRate = 0.02
    nEpochs = 4000
    nInputs = 10
    nHidden = 8
    nOutputs = 6

    weightsHidden = 2 * (np.random.rand(nHidden, nInputs) - 0.5)
    weightsOutput = 2 * (np.random.rand(nOutputs, nHidden) - 0.5)

    d_w1_old = np.zeros((nHidden, nInputs))
    d_w2_old = np.zeros((nOutputs, nHidden))
    for n in range(nEpochs):
        meanSquareError = 0
        for i in range(len(trainSet[:, 0])):
            # calculate final output
            hidden_o = np.dot(weightsHidden, trainSet[i, :])
            hidden_o = sigmoid(hidden_o)
            outputs = np.dot(weightsOutput, hidden_o)
            outputs = sigmoid(outputs)

            # determine error and update weights
            dk = calcError(outputs, trainAns[i, :])
            mu_vec = hidden_o * (1 - hidden_o) * np.dot(np.transpose(weightsOutput), dk)
            d_w1_new = +lRate * np.outer(mu_vec, trainSet[i, :])
            d_w2_new = +lRate * np.outer(dk, hidden_o)
            weightsHidden += d_w1_new + momentum * d_w1_old
            weightsOutput += d_w2_new + momentum * d_w2_old
            d_w1_old = d_w1_new.copy()
            d_w2_old = d_w2_new.copy()

        #  calculate MSE after each epoch and break if below error threshold break
        for i in range(len(validSet[:, 0])):
            hidden_o = np.dot(weightsHidden, validSet[i, :])
            hidden_o = sigmoid(hidden_o)
            outputs = np.dot(weightsOutput, hidden_o)
            outputs = sigmoid(outputs)
            meanSquareError += (np.sum(outputs - validAns[i, :])) ** 2
        if (meanSquareError / len(validSet[:, 0]) < 0.001):
            break
    return weightsHidden, weightsOutput


def confusionMatrix(testInputs, weightsHidden, weightsOutput, answer):
    """
    Calculates the confusion matrix for a test run. This is the function that performs the testing after training
    is completed.
    :param test_inputs: Input data to the network
    :param weightsHidden: Weights between input and hidden layer
    :param weightsOutput: Weights between hidden layer and output
    :param answer: Known answers for each data input
    :return: Confusion matrix, in separate scalar quantities
    """
    rows = len(testInputs[:,0])
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    
   # Run through all expected outputs and calculated outputs and determine the Confusion Matrix
    for i in range(rows):
        hiddenOutputs = np.dot(weightsHidden, testInputs[i,:])
        hiddenOutputs = sigmoid(hiddenOutputs)
        outputs = np.dot(weightsOutput, hiddenOutputs)
        outputs = sigmoid(outputs)

        # Short way to quickly make the output vector for a single row at a time
        outputs = [1 if elem == np.amax(outputs) else 0 for elem in outputs]

        # Now check for True Positives/Negatives and False Positives/Negatives
        for j in range(len(outputs)):
            if outputs[j] == answer[i,j]:
                if answer[i,j] == 1:
                    truePos += 1
                else:
                    trueNeg += 1
            else: 
                if outputs[j] == 1:
                    falsePos += 1
                else:
                    falseNeg += 1
                    
    return truePos, falsePos, trueNeg, falseNeg


def sigmoid(x):
    """
    Evaluates a sigmoid function at the point x
    :param x: Input point
    :return: Sigmoid evaluated at the point x
    """

    sigmoid = 1. / (1. + np.exp(-x))
    return sigmoid


def sigmoidDerivative(x):
    """
    Calculates the derivative for a sigmoid function at a point
    :param x: Point to evaluate at
    :return: Derivative of a sigmoid evaluated at the point x
    """

    sigmoid = 1. / (1. + np.exp(-x))
    return sigmoid * (1. - sigmoid)


def calcError(outputs, answer):
    """
    Error calculation between expected output and calculated output
    :param outputs:
    :param answer:
    :return: Error between the expected output and calculated outputs
    """
    return (answer - outputs) * outputs * (1 - outputs)
           
        