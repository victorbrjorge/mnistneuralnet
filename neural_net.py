#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import random
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.metrics import log_loss
from scipy.special import expit
import sys

class Neuron(object):

    def __init__ (self, n_inputs):
        self.n_inputs = n_inputs

        self.weights = np.array([random.uniform(-0.001, 0.001) for i in range(n_inputs)], dtype='float64')
        self.bias = random.uniform(-0.001, 0.001)

        self.delta = 0.0

        self.future_delta_w = np.zeros(n_inputs, dtype='float64')
        self.future_delta_b = 0.0
        self.batch_size = 0

        self.output = 0.0

    def activate(self, inputs):
        a = np.dot(self.weights, inputs) + self.bias # w*x+b
        self.output = expit(a) #sigmoid implemented by scipy deals better with over/underflow errors and it's faster
        return self.output

    def get_output(self):
        return self.output

    def solve_delta(self, expected_output):
        self.delta = self.output - expected_output
    
    def get_delta(self):
        return self.delta

    def update_weights_batch(self, inputs, l_rate, update = False):
        
        self.future_delta_w = np.add(self.future_delta_w, inputs * self.delta) 
        self.future_delta_b += self.delta
        self.batch_size+=1
        
        if update:
            self.weights = np.subtract(self.weights, (self.future_delta_w / self.batch_size) * l_rate)
            self.bias -= (self.future_delta_b / self.batch_size) * l_rate
            
            self.future_delta_w = np.zeros(self.n_inputs, dtype='float64')
            self.future_delta_b = 0.0
            self.batch_size = 0

    def update_weights_sgd(self, inputs, l_rate):
        self.weights = np.subtract(self.weights, inputs * self.delta * l_rate) 
        self.bias -= self.delta * l_rate 

    def get_weights(self):
        return self.weights

class NeuralNetwork(object):
    
    def __init__(self, layers_size = None):
        
        self.network = list()
        
        self.n_classes = layers_size[-1]
        self.layers_size = layers_size
        self.n_layers = len(layers_size)
        
        for i in range(1, self.n_layers): #input layer is just theorical
            
            n_inputs = layers_size[i-1]
            n_neurons = layers_size[i]

            self.network.append([Neuron(n_inputs) for i in range(n_neurons)])

    def propagate(self, example, verbose=False):

        for layer in self.network:
            next_input = []
            for neuron in layer:
                next_input.append(neuron.activate(example))
            example = np.array(next_input, dtype='float64')
            if verbose:
                print example
    
        return example

    def backpropagate (self, expected_output):
        
        for i in reversed(range(len(self.network))):
            
            layer = self.network[i]
             
           if i == len(self.network) - 1: #if it is the output layer
                j = 0
                for neuron in layer:
                    neuron.solve_delta(expected_output[j])
                    j+=1
            else:
                for k in range(len(layer)):
                    
                    prop_error = 0.0
                    neuron = layer[k]
                    
                    next_layer = self.network[i+1]
                    for next_neuron in next_layer:
                        prop_error += (next_neuron.get_weights()[k] * next_neuron.get_delta())

                    neuron.solve_delta(prop_error)

    def update_weights_sgd(self, inputs, l_rate):
        for i in range(self.n_layers - 1):
            layer = self.network[i]
            if i != 0:
                inputs = np.array([neuron.get_output() for neuron in self.network[i-1]], dtype='float64')
            for neuron in layer:
                neuron.update_weights_sgd(inputs, l_rate)

    def update_weights_batch(self, inputs, l_rate, update):
        for i in range(self.n_layers - 1):
            layer = self.network[i]
            if i != 0:
                inputs = np.array([neuron.get_output() for neuron in self.network[i-1]], dtype='float64')
            for neuron in layer:
                neuron.update_weights_batch(inputs, l_rate, update)

    def train_SGD(self, data, labels, l_rate, epochs):
        for epoch in range(epochs):
            j = 1
            error = 0.0
            right = 0.0
            predicts = list()
            
            for row in data:
                expected_output = [0.0 for i in range(self.n_classes)]
                expected_output[labels[j-1]] = 1.0
            
                inputs = np.array(row, dtype='float64')

                output = self.propagate(inputs, verbose=False)
                error += self.cost_function(output, expected_output)
                
                self.backpropagate(expected_output)
                self.update_weights_sgd(inputs, l_rate)

                predicts.append(output.argmax())
                
                if output.argmax() == labels[j-1]:
                    right+=1.0
                j+=1        
            print('>epoch=%d \t lrate=%.3f \t error=%.3f \t accuracy=%.2f' % (epoch, l_rate, error/j, right/j))

    def train_batch(self, data, labels, l_rate, epochs, batch_size):
        
        for epoch in range(epochs):
            j = 1
            error = 0.0
            right = 0.0
            predicts = list()
            
            for row in data:
                
                expected_output = [0.0 for i in range(self.n_classes)]
                expected_output[labels[j-1]] = 1.0
            
                inputs = np.array(row, dtype='float64')

                output = self.propagate(inputs, verbose=False)
                error += self.cost_function(output, expected_output)
                
                self.backpropagate(expected_output)
                self.update_weights_batch(inputs, l_rate, update=False) # we save the value of the new weight but we dont really update it
                
                if j % batch_size == 0:
                    self.update_weights_batch(inputs, l_rate, update=True)

                predicts.append(output.argmax())
                
                if output.argmax() == labels[j-1]:
                    right+=1.0
                j+=1

            print('>epoch=%d \t lrate=%.1f \t error=%f \t accuracy=%.3f' % (epoch, l_rate, error/j, right/j))
 

    def predict(self, row):
        predicted = self.propagate(row, verbose=False)
        return predicted.argmax()

    #Cross-entropy cost function from sklearn
    def cost_function(self, output, expected_output):
        return log_loss(expected_output, output, normalize=False)

def main():
    # TO RUN: ./backpropagation.py gd 50 1 20
    algorithm = sys.argv[1]
    hidden_layer_size = int(sys.argv[2])
    l_rate = float(sys.argv[3])
    epochs = int(sys.argv[4])

    layers_size = [784, hidden_layer_size, 10]

    net = NeuralNetwork(layers_size)

    train_file = open('data_tp1')
    reader = csv.reader(train_file)

    data = []
    labels = []

    for line in reader:
        data.append([float(x) for x in line[1:]])
        labels.append(int(line[0]))

    train_file.close()

    normalizer = Normalizer()
    data = normalizer.fit_transform(data)

    if algorithm == 'minibatch':
        batch_size = int(sys.argv[5])
        net.train_batch(data, labels, l_rate, epochs, batch_size)
    elif algorithm == 'gd':
        batch_size = len(data)
        print batch_size
        net.train_batch(data, labels, l_rate, epochs, batch_size)
    else:
        net.train_SGD(data, labels, l_rate, epochs)

if __name__ == '__main__':
    main()