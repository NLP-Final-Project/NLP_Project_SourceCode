import numpy as np
import pandas as pd

#importing Data
dataset = pd.read_csv('Vector_Assay_1.csv')

X1 = np.asarray(dataset)
# sigmoid function
def nonlin(x,deriv=False): 
    if(deriv==True): 
        return x*(1-x) 
    return 1/(1+np.exp(-x)) 

# input dataset
X = X1[:100,1:101]
print(X)

# output dataset 
y = X1[100:102,1:101].T
print(y)
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((100,2)) - 1

for iter in range(10000): 
    
    # forward propagation 
    l0 = X 
    l1 = nonlin(np.dot(l0,syn0)) 
    
    # how much did we miss? 
    l1_error = y - l1 
    
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1 
    l1_delta = l1_error * nonlin(l1,True) 
    
    # update weights 
    syn0 += np.dot(l0.T,l1_delta)
    
print("Output After Training:")
print(l1)
