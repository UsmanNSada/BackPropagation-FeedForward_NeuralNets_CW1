# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:52:36 2020

@author: UTHMAN
"""
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return(1/(1 + np.exp(-x))) # TODO
def sigmoid_d(x):
    return(sigmoid(x))*(1-sigmoid(x)) # TODO
def relu(x):
    return(max(0,x)) # TODO
def relu_d(x):
    return(1 if x > 0 else 0) # TODO
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,10,10,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape] #Defining z
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
        self.j = 0
        self.m = 0
        # Our variables
        
        #print(self.z[4])
        #print(self.z[4].shape)
        
   
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = (x/255) - 0.5      # Center the input values between [-0.5,0.5]
        #print(self.a[0])
        
        for i in range(10):
            self.z[1][i] = sum((self.a[0]) * (self.w[1][i]))
        self.z[1] = self.z[1] + self.b[1]    
        #print(self.z[1].shape)
        
        for i in range(10):
            self.a[1][i] = relu(self.z[1][i])
        #print(self.a[1])
        
        for i in range(10):
            self.z[2][i] = sum((self.a[1]) * (self.w[2][i])) 
        self.z[2] = self.z[2] + self.b[2]
        
        for i in range(10):
            self.a[2][i] = relu(self.z[2][i])
        #print(self.a[2])
        
        #for i in range(10):
            #self.z[3][i] = sum((self.a[2]) * (self.w[3][i]))
        #self.z[3] = self.z[3] + self.b[3]
        
        
        #for i in range(10):
            #self.a[3][i] = relu(self.z[3][i])
        #print(self.a[3])
        
        for i in range(10):
            self.z[self.L-1][i] = sum((self.a[self.L-2]) * (self.w[self.L-1][i]))
        self.z[self.L-1] = self.z[self.L-1] + self.b[self.L-1]
        
        for i in range(10):
            self.a[self.L-1][i] = np.exp(self.z[self.L-1][i]) 
        self.a[self.L-1] = ((self.a[self.L-1])/np.sum(self.a[self.L-1]))
        #self.a[4] = softmax(self.z[4])
        # TODO
        
        return(self.a[self.L-1])
    
    def softmax(self, z):
        Q = 0
        for i in range(10):
            Q += np.exp(self.z[self.L-1][i])
        
        for m in range(10):
            self.a[self.L-1][m] = np.exp(self.z[self.L-1][m])/Q
        return(self.a[self.L-1])

    def loss(self, pred, y):
        class_index = np.argmax(y)
        print(class_index)
        Pj = pred[class_index] 
        return(-np.log(Pj))
        
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        self.delta[self.L-1] = self.a[self.L-1] - y
        relz3=np.zeros(10,)
        relz2=np.zeros(10,)
        relz1=np.zeros(10,)
        
        #for i in range(10):
            #relz3[i] = relu_d(self.z[3][i])
        for i in range(10):
            relz2[i] = relu_d(self.z[2][i])
        for i in range(10):
            relz1[i] = relu_d(self.z[1][i])
        
        #self.delta[3] = np.multiply((np.dot((self.w[4].T),(self.delta[4]))), relz3)
        self.delta[2] = np.multiply((np.dot((self.w[3].T),(self.delta[3]))), relz2)
        self.delta[1] = np.multiply((np.dot((self.w[2].T),(self.delta[2]))), relz1)
     
        self.dw[self.L-1] = np.dot(self.delta[self.L-1].reshape(10,1),self.a[self.L-2].reshape(1,10))
        #self.dw[3] = np.dot(self.delta[3].reshape(10,1),self.a[2].reshape(1,10))
        self.dw[2] = np.dot(self.delta[2].reshape(10,1),self.a[1].reshape(1,10))
        self.dw[1] = np.dot(self.delta[1].reshape(10,1),self.a[0].reshape(1,784))
        
    
        for l in range(self.L):
            self.db[l] = self.delta[l]   
        
        # 5 layers
        
        return(self.delta[3],self.delta[2], self.delta[1], self.db[1], self.db[2], self.db[3], self.dw[1], self.dw[2], self.dw[3])
    
    # Return predicted image class for input x
    def predict(self, x):
        
        self.j = np.argmax(self.forward(x))
        self.m = (self.forward(x))[self.j]
        return(self.j)

    # Return predicted percentage for class j
    def predict_pct(self, j):
        
        
        #class_index = np.argmax(j)
        #print(class_index)
        #Pj = pred[class_index]  
        return self.m
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=100,
            epsilon=0.01,
            epochs=1000):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                for l in range(self.L):
                    self.delta[l] = np.zeros
                    self.dw[l] = np.zeros
                    self.db[l] = np.zeros
                # TODO
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    # TODO
                    self.forward(x)
                    
                    # Compute gradients
                    self.backward(x, y)
                    # TODO

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] = self.w[l] - (epsilon*(self.dw[l]))
                    self.b[l] = self.b[l] - (epsilon*(self.db[l]))
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)


# Start training with default parameters.

def main():
    bp = BackPropagation()
    #bp.predict(bp.trainX[0])
    #bp.predict_pct(bp.j)
    #bp.forward(bp.trainX[0])
    #bp.loss(bp.a[bp.L-1], bp.trainY[0])
    #bp.backward(bp.trainX[0],bp.trainY[0])
    bp.sgd()

if __name__ == "__main__":
    main()
    
