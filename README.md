# mygrad
This repository implements a multi-layer perceptron with the backpropagation algorithm from scratch.

## Introduction

As a physicist, despite being employed in deep learning for a long time, I always took a handwaving approach to the backpropagation algorithm because I am so familiar with the multivariable chain rule. But that changed today. I coded this repositrory by following Andrej Karpathy's recent [Youtube tuitorial](https://www.youtube.com/watch?v=VMj-3S1tku0&ab_channel=AndrejKarpathy) explaining the working of his [micrograd](https://github.com/karpathy/micrograd) library. Right now, the code look's very similar to micrograd, but I intend to develop it further for learning purposes as time allows.

## Backpropagation example for a simple neuron

Propagation of gradient for a simple neuron for an input $[1, 1]$.

![Neuron_backpropagation](https://user-images.githubusercontent.com/43025445/187179073-a1994eb2-2f00-4078-b467-485aca12bcc5.svg)

## Training the Multilayer Perceptron 

### Model

We choose a multilayer perceptron with a two hidden layers of size $8$ with $ReLU$ activation functions for all layers.

### Training data

The ```make_moons``` function from ```sklearn``` was used to create the following simple training dataset of $100$ points.

![Training_data](https://user-images.githubusercontent.com/43025445/187179484-1a2f6220-6628-4f9e-8c6a-a956915f3884.png)

The model before training has the following decision bounday,

![Model_predictions_before_training](https://user-images.githubusercontent.com/43025445/187179571-410e1417-8db2-46c3-8ba8-e6d67a5ddb60.png)

It later improves to correctly seperate the classes after training for $100$ epochs with a flat learning rate of $0.01$, mean squared loss, and $L2$ regularization ( $\alpha = 0.0001$ ).

![Model_predictions_after_training](https://user-images.githubusercontent.com/43025445/187179683-620ef94b-b642-43b4-81f4-56952286520a.png)

### Loss and accuracy across the epochs

![Training_loop](https://user-images.githubusercontent.com/43025445/187179769-90383e54-3e1d-4d22-b657-36e7ffc75fbc.png)

## Future Goals

1. Extend the `Value` class to a `Tensor` class to natively support matrix operations.
2. Add convolutional filters and recurrent neural networks.
3. Attempt MNIST dataset.
