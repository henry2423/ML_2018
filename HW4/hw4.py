import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def forward_propagate(X, theta1, theta2):
    m = X.shape[0] # number of data instances
    
    #Write codes here
    #Layer1
    extraBias1 = np.ones((m,1))
    a1 = np.append(extraBias1, X, axis=1) #append bias on every data instances
    z2 = np.dot(a1, theta1.transpose())
    # Layer2
    extraBias2 = np.ones((z2.shape[0],1))
    a2 = np.append(extraBias2, sigmoid(z2), axis=1)
    z3 = np.dot(a2, theta2.transpose())
    # Output
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h
    
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m): ##calculate error on every instance
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    return J
    
# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
# unravel the parameter array into parameter matrices for each layer
#theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
#theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

# a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    
    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    delta1 = np.zeros(theta1.shape)  # (10, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 11)
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    #print(J)  ## print cost for every iterate 
    
    # perform backpropagation
    for i in range(m):  ## update weight by every instance
        a1Del = a1[i,:] #(1, 401)
        z2Del = z2[i,:] #(1, 10)
        a2Del = a2[i,:] #(1, 11)
        hDel = h[i,:] #(1, 10)
        yDel = y[i,:] #(1, 10)
        
        grad3Del = hDel - yDel # (1, 10)
        
        z2Del = np.append(np.ones((1,1)), z2Del, axis=1)  # (1, 11)
        temp = np.dot(theta2.transpose(), grad3Del.transpose()) # (11, 1)
        grad2Del = np.multiply(temp.transpose(), sigmoid_gradient(z2Del))  # (1, 11)

        delta1 = delta1 + np.dot((grad2Del[:,1:]).transpose(), a1Del) ## remove grad2Del[:,0] #(10, 401)
        delta2 = delta2 + np.dot(grad3Del.transpose(), a2Del) #(10, 11)
        
    delta1 = delta1 / m
    delta2 = delta2 / m

    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    gradient = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, gradient
    
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250})
      
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))