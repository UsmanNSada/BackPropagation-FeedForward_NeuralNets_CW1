
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
    return(1/(1+ np.exp(-x)))# TODO
def sigmoid_d(x):
    return(sigmoid(x))*(1-sigmoid(x))
     # TODO
def relu(x):
    return(max(0,x)) # TODO
def relu_d(x):
    return(1 if x > 0 else 0) 
    # TODO
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)
        

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]

        # Custom
        self.shape = network_shape
        #self.pred = np.zeros(10) 
        
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = (x/255) - 0.5      # Center the input values between [-0.5,0.5]
        #print(self.w[1].shape)
        #print(self.delta[4])
        #print(self.delta[4].shape)
        #self.z[0] = np.multiply(self.w[0], self.a[0]) + self.b[0]
        #print(self.z[0].shape)
        #for i in range(784):
            #self.z[0][i] = sum((self.a[0][i]) * (self.w[0])) + self.b[0][i]
        #print(self.z[0])
        #print(self.z[0].shape)
        
        #zi = np.zeros(784)
        #for i in range(784):
            #zi[i] = sigmoid(self.z[0][i])
        #print(zi)
        
        #print(zi.shape)
        #print(self.z[1].shape)
        #self.a[1] = np.dot(self.w[1], zi)
        #for i in range(20):
            #self.a[1][i] = sigmoid(self.a[1][i])
        self.z[1] = np.dot(self.w[1], self.a[0]) + self.b[1]
        print(self.z[1])
        
        for i in range(len(self.z[1])):
            self.a[1][i] = sigmoid(self.z[1][i])
        #print(self.a[2])
        self.z[2] = np.dot(self.w[2], self.a[1]) + self.b[2]
        #print(self.z[2])
        
        for i in range(len(self.z[2])):
            self.a[2][i] = sigmoid(self.z[2][i])
        #print(self.a[3])
        #print(self.z[3].shape)
        self.z[3] = np.dot(self.w[3], self.a[2]) + self.b[3]
        #print(self.z[3])
        
        #zi1 = np.zeros(20)
        #for i in range(20):
            #zi1[i] = sigmoid(self.z[3][i])   
        #print(zi1)
        for i in range(len(self.z[3])):
            self.a[3][i] = sigmoid(self.z[3][i])
        
        self.z[4] = np.dot(self.w[4], self.a[3]) + self.b[4]
        #print(self.z[4])
        for i in range(len(self.z[4])):
            self.a[4][i] = sigmoid(self.z[4][i]) 
        
        for i in range(10):
            self.a[4][i] = np.exp(self.a[4][i]) 
        self.a[4] = ((self.a[4])/np.sum(self.a[4]))
        #print(self.a[4])
        # TODO
        #return(self.a[self.L-1])

    def softmax(self, z):
        # TODO 
        self.z[4] = np.exp(self.z[4])
        #return self.z[4] / np.sum(self.z[4])
        
    def loss(self, pred, y):
        print(pred)
        print(self.a[4])
        return(- np.log(pred*y[np.argmax(y)]))
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        #print(y) # to check the one hot encoding 
        self.delta[4] = self.a[4] - y # computes the local gradient of the softmax layer
        #print(self.delta[4])
        
        rel = np.zeros(20)
        for i in range(20):
            rel[i] =  sigmoid_d(self.z[3][i]) # computes the derivative of the relu function at z[3]
        #print(self.delta[3].shape)
        self.delta[3] = np.multiply((rel), (np.dot((self.w[4].T),(self.delta[4])))) # computes the local gradient of the hidden layer 4
        #print(self.delta[3].shape)
        
        rel1 = np.zeros(20)
        for i in range(20):
            rel1[i] =  sigmoid_d(self.z[2][i]) # computes the derivative of the relu function at z[2]
        #print(self.delta[2].shape)    
        self.delta[2] = np.multiply((rel1), (np.dot((self.w[3].T),(self.delta[3]))))  # computes the local gradient of the hidden layer 3
        #print(self.delta[2].shape)
        
        rel2 = np.zeros(20)
        for i in range(20):
            rel2[i] =  sigmoid_d(self.z[1][i])  # computes the derivative of the relu function at z[1]
        
        #print(self.delta[2].shape)
        self.delta[1] = np.multiply((rel2), (np.dot((self.w[2].T),(self.delta[2]))))  # computes the local gradient of the hidden layer 2
        #print(self.delta[1].shape)
        
        #print(self.delta[4].shape)
        #print(self.delta[3].shape)
        #print(self.dw[3].shape)
        self.dw[4] = np.dot(self.delta[4].reshape(10,1), self.a[3].reshape(1,20)) # computes the partial derivative w. r. t the weights at layer 5
        #.reshape(10,1)
        #.reshape(1,20)
        #print(self.dw[4].shape)
        self.dw[3] = np.hstack((self.delta[3], self.a[2]))  # computes the partial derivative w. r. t the weights at layer 4
        #print(self.dw[3].shape)
        
    
        #(self.a[1].shape)
     
        #np.resize(self.a[1], (1,20))
        #print(self.a[1].shape)
        #self.dw[2] = (self.a[1].reshape(20,1)).dot(self.delta[2].reshape(1,20))
        #self.dw[3]= (self.a[2].reshape(20,1)).dot(self.delta[3].reshape(1,20))
        #print(self.dw[3])
       
        #print(self.a[1])
        
        #print(self.a[1])
        self.dw[2] = np.dot(self.delta[2].reshape(20,1), self.a[1].reshape(1,20))  # computes the partial derivative w. r. t the weights at layer 3
        #print(self.dw[2].shape)
    
        self.dw[1] = np.dot(self.delta[1].reshape(20,1), self.a[0].reshape(1,784))  # computes the partial derivative w. r. t the weights at layer 2
        #print(self.dw[1].shape)
        
        #print(self.dw[0].shape)  # computes the partial derivative w. r. t the biases in all the layers
        for l in range(self.L):
            self.db[l] = self.delta[l]
  
        #print(self.db[4])    # print the partial derivative w. r. t. the biases at layer 5 (the softmax layer)
        

        return(self.delta[1], self.delta[2], self.delta[3], self.delta[4], self.db[1], self.db[2], self.db[3], self.db[4], self.dw[1], self.dw[2], self.dw[3],self.dw[4])
        # TODO

    # Return predicted image class for input x
    def predict(self, x):
        return(np.argmax(self.a[4]))
    # TODO

    # Return predicted percentage for class j
    def predict_pct(self, j):
        for i in range(10):
            jClass = self.a[4][i] 
      
        #print(jClass)
        return(jClass)
                # TODO 
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    def sgd(self,
            batch_size=50,
            epsilon=0.55,
            epochs=10000):

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
                self.batch_a = [np.zeros(m) for m in self.shape]
                
                self.w = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
                
                loss_log = []
                test_acc_log= [] 
                train_acc_log = []
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
                    # TODO
                    self.backward(x, y)
                    self.predict(x)
                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                  
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                   
                    
                    self.w[l] = self.w[l] - (epsilon*(self.dw[l]))
                    # TODO
                    self.b[l] =  self.b[l] - (epsilon*(self.db[l]))
                    # TODO
                     
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
    
    #print(bp.forward(bp.trainX[1]))
    #(bp.loss(bp.a[bp.L-1], bp.trainY[1]))
    #bp.backward(bp.trainX[1],bp.trainY[1])
    bp.sgd()
    
if __name__ == "__main__":
    main()
    
