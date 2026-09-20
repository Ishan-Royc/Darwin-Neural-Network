# Darwin-Neural-Network
A Neural Network written in TypeScript which trains using Darwinian selection of the 'fittest' networks (i.e. those that match a given transformation of a set of input elements the best).

This particular repo is meant to be a proof-of-concept for this method of training them, since the actual network itself isn't very sophisticated and probably doesn't even resemble real-world neural networks in the way it works. The main thing I was trying to prove with this project is that hypothetically, a neural network can be trained through a Darwinian 'selection' process, rather than through the standard training methods, and that a reasonably decent end-result can be produced.

## The Problem
Before I describe how the model is trained, I'll start by describing the general type of neural network my program generates and what it tries to do. In essence, I'm feeding the network an input array and expecting it to perform a certain mathematical transformation to the elements to get an output array. I've provided the program with a list of primitive functions that are combined into randomly-generated composite functions that are then assigned to each node. Nodes are connected by connections, which have a certain weight (which is multiplied with the output of the previous node before being passed on to the next node). Each node also has a bias, which is added to its output before being sent to its output connection.

## Training
