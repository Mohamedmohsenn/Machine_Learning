import itertools
import pandas as pd
import numpy as np
import random

def getCompination(x):
    resultSet = []
    for i in range(1,len(x)+1):
        resultSet.append(list(itertools.combinations(x, i)))
    item = []
    for i in resultSet:
        for j in i:
            item.append(list(j))
    return item            


def normalize(X):  
    norm_X = (X - X.mean())/(X.std())
    return norm_X

def splitData(ds):
    shuffle_df = ds.sample(frac=1)
    train_size = int(0.75 * len(ds))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return train_set,test_set


def prediction(w,x,b) :
    h = np.dot(x,w.T)+b
    for i in range(len(h)) :
        if h[i] > 0 :
            h[i] = 1
        elif h[i] < 0 :
            h[i] = -1
    return h
    

    
def gredientDesc(alpha,w,x,y,iterations,b,lamda) : 
    for i in range(iterations) :
        for j in range (y.shape[0]) :
            if ((np.dot(x[j],w.T)+b) * y[j]) >= 1:
                w = w - (alpha*2*lamda*w)
            else :    
                w = w + (alpha *( (y[j] * x[j]) - (2 * lamda * w)))
                b = b + (alpha * y[j])
    return w,b
    
def calcAccuracy(h,y):
    right = 0
    wrong = 0

    for i in range(y.shape[0]):
        if (y[i] == h[i]):
            right+=1
        else:
            wrong+=1
    calcAccuracy = right/y.shape[0]*100
    return calcAccuracy  


def updateY(y) :
   return np.where(y==0 , -1, 1)



dataset = pd.read_csv("heart.csv")


dataset['target'] = updateY(dataset['target'])


X = dataset.drop(['target'], axis=1)
lamda = 0.001
b = 0
trainSet,testSet = splitData(dataset)

yTrain = trainSet['target']
yTrain = np.expand_dims(yTrain,axis=-1)


yTest = testSet['target']
yTest = np.expand_dims(yTest,axis=-1)


mx = []
wMax = []
bMax = 0
mxAccuracy = 0

z = getCompination(X)
random.shuffle(z)

for i in range (20) :
    xTrain = trainSet[z[i]]
    xTrain = normalize(xTrain) 
    xTrain = np.array(xTrain)
    
    xTest = testSet[z[i]]
    xTest = normalize(xTest)
    xTest = np.array(xTest)
    
    w = np.zeros(xTest.shape[1])
    w = np.array(w).reshape(1,xTest.shape[1])
    
    w,b = gredientDesc(0.01,w,xTrain,yTrain,500,b,lamda)
    
    h = prediction(w,xTest,b)
    
    acc = calcAccuracy(h,yTest)
    
    if acc > mxAccuracy :
        mxAccuracy = acc
        mxIndex = i
        mx = z[i]
        wMax = w
        bMax = b
        

print("bestFeatures on first 20 of the shuffle : ",mx)
print("Max Accuracy : ",mxAccuracy)




print("--------------------------------------")


trainSet,testSet = splitData(dataset)

yTrain = trainSet['target']
yTrain = np.expand_dims(yTrain,axis=-1)


yTest = testSet['target']
yTest = np.expand_dims(yTest,axis=-1)

xTrain = trainSet[["cp","thalach","slope"]]
xTrain = normalize(xTrain) 
xTrain = np.array(xTrain)


xTest = testSet[["cp","thalach","slope"]]
xTest = normalize(xTest)    
xTest = np.array(xTest)

acc = []

alphaArr = [0.001,0.003,0.01,0.03,0.1,0.3,0.5]


for i in alphaArr :
    w = np.array([0,0,0]).reshape(1,3)
    b = 0
    w,b = gredientDesc(i,w,xTrain,yTrain,500,b,lamda)
    h = prediction(w,xTest,b)
    acc.append(calcAccuracy(h,yTest))
    
for i in range(len(alphaArr)) :
    print("Accuracy of ",alphaArr[i]," = ",acc[i])






























            




