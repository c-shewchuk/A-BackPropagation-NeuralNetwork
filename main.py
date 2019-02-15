import numpy as np
from random import seed
from random import random
import backprop as bp

seed(1)
training_data = np.genfromtxt('training.csv', delimiter=',')
test_data = np.genfromtxt('test.csv', delimiter=',')
for layer in training_data:
    print(layer)
print('New Weights\n')
network = bp.initNetwork(9,8,6)
bp.train_network(network, training_data, 0.5, 1000, 6)

for layer in network:
    print(layer)

for row in test_data:
    prediction  = bp.predict(network, row)
    print('Expected=%d, Got=%d' %(row[-1], prediction))


