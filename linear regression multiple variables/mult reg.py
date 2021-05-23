import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt


def splitData(ds):
    shuffle_df = ds.sample(frac=1)
    train_size = int(0.7 * len(ds))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return train_set,test_set

def normalize(x): 
    norm_x = (x - x.mean())/(x.std())
    return norm_x

def hypothesisFunction(theta,x):
   return np.dot(x,theta.T)


def calcError(theta,x,y) :
    h = hypothesisFunction(theta,x)
    return (1 / (2*y.size)) * np.sum(np.power((y-h),2))


def gredientDesc(alpha,theta,x,y,iterations) :
    m = y.size       
    errors = []
    for i in range(iterations) :
        h = hypothesisFunction(theta,x)
        theta = theta - (alpha/m * np.dot((h-y).T,x))
        errors.append(calcError(theta,x,y))
    return theta,errors

dataSet = pd.read_csv("house_data.csv")

trainSet,testSet = splitData(dataSet)
yTrain = trainSet["price"]
yTrain = np.expand_dims(yTrain,axis=-1)

xTrain = trainSet[["grade","bathrooms","lat","sqft_living","view"]]
xTrain = normalize(xTrain)
xTrain = np.hstack((np.matrix(np.ones(xTrain.shape[0])).T, xTrain)) 

yTest = testSet["price"]
yTest = np.expand_dims(yTest,axis=-1)

xTest = testSet[["grade","bathrooms","lat","sqft_living","view"]]
xTest = normalize(xTest)
xTest = np.hstack((np.matrix(np.ones(xTest.shape[0])).T, xTest)) 


theta = np.array([0,0,0,0,0,0]).reshape(1,6)


alphaArr = [0.001,0.003,0.01,0.03,0.1,0.3,0.5]


iterations = 1500
for i in alphaArr :
    theta,errors = gredientDesc(i,theta,xTrain,yTrain,iterations)
    h = hypothesisFunction(theta,xTest)
    print('-----------------------------------')
    print("alpha :")
    print(i,"\n")
    plt.plot(list(range(iterations)),errors)
    plt.show()    
    #print("errors :")
    #print(errors,"\n")
    print("Accuracy of ",i," is :")
    print(sm.r2_score(yTest, h)*100)
    theta = np.array([0,0,0,0,0,0]).reshape(1,6)