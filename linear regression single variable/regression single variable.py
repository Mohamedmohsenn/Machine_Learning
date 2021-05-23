import numpy as np
#import pandas as pd
import os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
#import matplotlib.pyplot as plt
#from collections import Counter
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#import seaborn as sn
#from sklearn.metrics import accuracy_score, recall_score
 
def splitData(ds):
    shuffle_df = ds.sample(frac=1)
    train_size = int(0.7 * len(ds))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return train_set,test_set


def normalize(X):  
    norm_X = (X - X.mean())/(X.std())
    return norm_X

def calcError(theta,x,y):
    h = theta[0] + theta[1]*x
    j = (1/(2*len(y))) * (sum((h-y)**2))
    return j
    

def greDescent(alpha,x,y,theta,itterationNumber) :
    s = y.size
    errors = []
    for i in range(itterationNumber) :
        h = theta[0] + (theta[1]*x)
        theta[0] = theta[0] - (alpha/s *sum(h-y))
        theta[1] = theta[1] - (alpha/s *sum((h-y)*x))
        errors.append(calcError(theta,x,y))
    return theta,errors
        
def hypothesisFunction(theta,x) :
    return theta[0]+theta[1]*x

data = pd.read_csv("house_data.csv")
trainSet,testSet = splitData(data)

xTrain = trainSet["sqft_living"]
xTrain = normalize(xTrain)
yTrain = trainSet["price"]

xTest = testSet["sqft_living"]
xTest = normalize(xTest)
yTest = testSet["price"]

theta = [0,0]


alphaArr = [0.001,0.003,0.01,0.03,0.1,0.3,0.5,1]

iterations = 1500
for i in alphaArr :
    print('-----------------------------------')    
    theta,errors = greDescent(i,xTrain,yTrain,theta,iterations)
    h = hypothesisFunction(theta,xTest)
    print("alpha :")
    print(i,"\n")
    plt.plot(list(range(iterations)),errors)
    plt.show()
    #print(errors,"\n")
    print("Accuracy of ",i," is :")
    print(sm.r2_score(yTest, h)*100)    
    theta = [0,0]
