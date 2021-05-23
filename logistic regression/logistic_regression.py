
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
def splitData(ds):
    shuffle_df = ds.sample(frac=1)
    train_size = int(0.7 * len(ds))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return train_set,test_set
"""

def normalize(X): 
    norm_X = (X - X.mean())/(X.std() )
    return norm_X


def calcHypothesis(theta,x) :  
    mult = np.dot(x,theta.T)
    h = 1 / (1 + np.exp(-mult))
    return h

def calcCostFunction(theta,x,y) :
    h = calcHypothesis(theta,x)
    y = np.squeeze(y)
    epsilon = 1e-7
    part1 = y*np.log(h+epsilon)
    part2 = (1-y)*np.log(1 - h+epsilon)
    r = -part1-part2
    return np.sum(r)
    
def gradDesc(alpha,theta,x,y):
    changable_cost = 1
    errors = []
    cost = calcCostFunction(theta,x,y)
    while changable_cost > .0001 :
        oldCost = cost
        errors.append(cost)
        mult = np.dot(x,theta.T)
        h = 1 / (1 + np.exp(-mult))
        theta = theta + (alpha* np.dot((y-h).T,x))
        cost = calcCostFunction(theta,x,y)
        changable_cost = oldCost - cost
    return theta,errors


def calcAccuracy(h,y):
    prediction = TransformHToPredictedValue(h)
    right=0
    wrong=0
    for i in range(len(prediction)):
        if prediction[i] == y[i]:
            right+=1
        else:
            wrong+=1
    calcAccuracy = right/len(y)*100
    return calcAccuracy  


def TransformHToPredictedValue(h):
    pred_value = np.where(h >= .5, 1, 0) 
    return np.squeeze(pred_value)


dataset = pd.read_csv("heart.csv")
y = dataset['target']
y = np.expand_dims(y,axis=-1)

x = dataset[['trestbps','chol','thalach','oldpeak']]
x = normalize(x)
x = np.hstack((np.matrix(np.ones(x.shape[0])).T, x)) 



"""
trainSet,testSet = splitData(dataset)
yTrain = trainSet['target']
yTrain = np.expand_dims(yTrain,axis=-1)

xTrain = trainSet[['trestbps','chol','thalach','oldpeak']]
xTrain = normalize(xTrain)
xTrain = np.hstack((np.matrix(np.ones(xTrain.shape[0])).T, xTrain)) 

yTest = testSet['target']
yTest = np.expand_dims(yTest,axis=-1)

xTest = testSet[['trestbps','chol','thalach','oldpeak']]
xTest = normalize(xTest)
xTest = np.hstack((np.matrix(np.ones(xTest.shape[0])).T, xTest)) 
"""


theta = np.array([0,0,0,0,0]).reshape(1,5)


alphaArr = [0.001,0.003,0.01,0.03,0.1,0.3,0.5,1]

z = []
for i in alphaArr :
    #theta,errors = gradDesc(i,theta,xTrain,yTrain)
    theta,errors = gradDesc(i,theta,x,y)
    #h = calcHypothesis(theta,xTest)
    h = calcHypothesis(theta,x)
    print('-----------------------------------')
    print("alpha")
    print(i,"\n")
    #print("errors :")
    #print(errors,"\n")
    print("Accuracy of ",i," is :")
    #acc = calcAccuracy(h, yTest)
    acc = calcAccuracy(h, y)
    z.append(acc)
    print(acc)
    theta = np.array([0,0,0,0,0]).reshape(1,5)

plt.plot(alphaArr,z)
plt.show()

















