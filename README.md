# CMPE 452 Assignment 2
## Backpropagating Neural Network
### Curtis Shewchuk
### SN: 10189026

## Instructions for Execution

To run the program, all you will need is Python 3.7 installed on your computer.
This can be installed from https://www.python.org/. Once installed, open up a terminal
(or command prompt on Windows), navigate to the directory where the code files are.
Make sure all files submitted are in this folder, they are all necessary. Once in this folder,
simply type `python main.py` and the program will run, train the network and produce
the 'outputs.txt' file. When a run completes, you should see in the terminal, "Run Complete". If an error occurs claiming that `numpy` is not installed, simply type in
the terminal (or command prompt) `pip install numpy` (I have had random issues with numpy
not being installed sometimes which is why I mention this). The outputs of each run
will be in the 'outputs.txt'. A sample is provided named 'outputs_bestrun.txt'. 


## Design Choices

#### Output Function
For the output function, a sigmoid function was used. The sigmoid was chosen for
the fact that it is differentiable everywhere, and that the function will always
output a value between 0 and 1 between negative infintity and positive infinity.

#### Learning rate, Termination Criteria and Momentum Constant
After many runs through the program, the learning rate was set at 0.02. This gave
reasonable results, including the best result as reported in 'outputs_bestrun.txt'. The 
terminating criteria chosen was the mean square error (MSE) approach. Each run, the updated
weights were tested on the validation set. If the MSE was below 0.001 on the validation set, then 
the program breaks. If this is never reached, than the training would end at the 5000 epochs.
The momentum parameter used was 1.

#### Network Configuration
The network uses 3 layers, one input, one hidden, and one output layer. The three layer
approach was chosen to minimize computation time when training the network. The network
has 9 input nodes, plus a bias for a total of 10 nodes. 8 hidden nodes were chosen as explained in class
that M-1 hidden layer nodes is the maximum required for the hidden layer (where M is number of inputs excluding bias).
Six output nodes were used, one for each glass type.

#### Regularization 
To regularize the data, the data sets were all normalized before being used by the network. The addition of
noise to the training data uses implicit regularization, which aids in training the data.


