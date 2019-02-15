import numpy as np
from random import random

def initNetwork(inputs, hidden, outputs):
    """
    Initialize the network. User defines the number of inputs, hidden layer nodes, and number
    of outputs.
    :param inputs: number of input layers
    :param hidden: number of hidden layer neurons
    :param outputs: number of output layers
    :return network: the created network, with original weights
    """
    network = list()
    hidden_layer = [{'weights':[random() for i in range(inputs + 1)]} for i in range(hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(hidden + 1)]} for i in range(outputs)]
    network.append(output_layer)
    return network

def weighted_sum(weights, inputs):
    """
    Creates a weighted sum of the inputs and the associated weights for the associated neuron
    :param weights: current layer weights
    :param inputs: inputs to the current layer
    :return: weighted sum of the inputs for that neuron
    """
    weighted_sum = weights[-1]
    for i in range(len(weights) -1):
        weighted_sum += weights[i] * inputs[i]
    return weighted_sum

def transfer_sigmoid(sum):
    """
    Transfer function in the neural network. Transfer function used is a sigmoid function
    :param sum: sum from the input neuron, and
    :return: returns the output of a sigmoid function, [0,1]
    """
    return 1.0/(1.0 + np.exp(-sum))

def sigmoid_derivative(output):
    """
    Returns the derivative of the output
    :param output: Used at the final layer to help back propagate error
    :return: derivative output
    """
    return output * (1.0 - output)

def forward_propagation(network, row):
    """
    Forward propagate the outputs of the nodes in the network, one layer at a time
    :param network: the network to use
    :param row: think it's the data row for input
    :return:
    """
    inputs = row
    for layer in network:
        new = []
        for neuron in layer:
            weight_sum  = weighted_sum(neuron['weights'], inputs)
            neuron['output'] = transfer_sigmoid(weight_sum)
            new.append(neuron['output'])
        inputs = new
    return inputs

def back_propagate_error(network, expected_output):
    """

    :param network:
    :param expected_output:
    :return:
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) -1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['sigma'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected_output[j] - neuron['output'])

        for k in range(len(layer)):
            neuron = layer[k]
            neuron['sigma'] = errors[k] * sigmoid_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    """

    :param network:
    :param row:
    :param l_rate:
    :return:
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i !=0:
            inputs = [neuron['output'] for neuron in network[i-1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['sigma'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['sigma']

def train_network(network, train, l_rate, n_epoch, num_outputs):
    """

    :param network:
    :param train:
    :param l_rate:
    :param n_epoch:
    :param num_outputs:
    :return:
    """
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagation(network, row)
            expected = [0 for i in range(num_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i] -outputs[i])**2 for i in range(len(expected))])
            back_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network, row):
    outputs = forward_propagation(network, row)
    return outputs.index((np.max(outputs)))


