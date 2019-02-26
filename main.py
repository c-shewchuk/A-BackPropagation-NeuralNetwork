"""
CMPE 452 Assignment 2
Backpropagating Neural Network
Curtis Shewchuk
SN: 10189026

This is the main file, which should be run for training and testing the network. There are two supporting
files, backprop.py and process.py. 'process.py' contains all the functions created to process the data. This
includes normalization, and the addition of noise (which are used in this script). 'backprop.py' contains the
functions which control the propagation of the

"""
import numpy as np
import backprop as bp
import process as pc

# GLOBAL CONSTANTS #
nInputs = 10
nHidden = 8
nOutputs = 6
momentum = 1
lRate = 0.02
nEpochs = 5000

# Import and initialize the data
filename = "GlassData.csv"
rawData = np.genfromtxt(filename, delimiter = ',', skip_header = 1)
rawData = rawData[:,1:len(rawData[0,:])] #Trim off the data point number in the GlassData.csv file
rows = len(rawData[:,0])
cols = len(rawData[0,:])

txtfile = "outputs.txt"
afile = open(txtfile,'w')


#Process and normalize the data
trainSet,trainAns,validSet,validAns,testSet,testAns = pc.processData(rawData, nOutputs)

# Add Noise
noise = 0.05
trainSet = pc.addNoise(trainSet,noise)

### DATA PROCESSING DONE AT THIS POINT ###

# Initialize ALL output nodes (hidden and final output) and the weights
hidden_o = np.zeros(nHidden)
outputs = np.zeros(nOutputs)

weightsHidden = 2*(np.random.rand(nHidden, nInputs)-0.5)
weightsOutput = 2*(np.random.rand(nOutputs, nHidden)-0.5)

## Call the train network function

weightsHidden, weightsOutput = bp.trainNetwork(trainSet, trainAns, validSet, validAns)


# Calculate the Confusion Matrix Values and Precision/Recall
truePos,falsePos,trueNeg,falseNeg = bp.confusionMatrix(testSet, weightsHidden, weightsOutput, testAns)

precision = truePos/(truePos + falsePos)
recall = truePos/(truePos + falseNeg)
num = len(testSet[:,0]) * 6

## Start writing to the text file
afile.write('CMPE 452 Assignment 2\nBackpropagating Neural Network\nCurtis Shewchuk\nSN:10189026\n\n')
afile.write('The initial weights matrices are given below:\n')
afile.write('Weights from input nodes to hidden nodes:\n')
afile.write("".join(str(elem) for elem in weightsHidden))
afile.write('\nWeights from hidden nodes to output nodes:\n')
afile.write(''.join(str(elem)for elem in weightsOutput))
afile.write('\n\n')


afile.write('Learning rate:' +str(lRate)+ '\n\n')
afile.write('Number of Epochs: '+str(nEpochs)+ '\n\n')
afile.write('Momentum Parameter: ' + str(momentum) + '\n\n')

# Write Final Values to the File 'outputs.txt'
afile.write('Total precision ' +str(precision)+ '\n\n')
afile.write('Total Recall: ' +str(recall)+ '\n\n')
afile.write('\n CONFUSION MATRIX\n')
afile.write('\n True Positive: '+ str(truePos*100/num)+ '% False Positive: ' + str(falsePos*100/num)+ '%\n')
afile.write('\n False Negative: ' + str(falseNeg*100/num)+ '% True Negative: ' + str(trueNeg*100/num)+ '%\n\n\n')

afile.write('Final Weight Matrices:\n')
afile.write('Input Layer to Hidden Layer:\n')
afile.write("".join(str(elem) for elem in weightsHidden))
afile.write('\n\nHidden Layer to Output Layer:\n')
afile.write(''.join(str(elem)for elem in weightsOutput))
afile.write('\n\n')

afile.close()

print('Run complete.')
