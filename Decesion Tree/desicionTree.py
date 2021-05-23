import pandas as pd
import numpy as np
import math
import statistics as sc

headers = ["goals","issue1","issue2","issue3","issue4","issue5","issue6","issue7","issue8","issue9","issue10","issue11","issue12","issue13","issue14","issue15","issue16"]

def getDataFrameForAttribute(x,attribute,header):    
    r = list(x[header])
    tmp = np.array(x)
    y = []
    
    for i in range (len(r)) :
        if r[i] == attribute : 
           y.append(tmp[i])
           
           
    y =  pd.DataFrame(y,columns = x.columns)
    
    y.drop([header],axis = 1 ,inplace = True)
    return y,list(y.columns)
    


def isPureFeature(x,y):
    goal = list(x['goals'])
    for i in goal :
        if i != y:
            return False
    return True
   

    
    
def getMaxInformationGain(x) :
    tmp = x["goals"].tolist()
    y = tmp.count('republican')
    n = tmp.count('democrat')
    probYesParent = y / (y+n)
    probNoParent = n / (y+n)
    entropyParent = -(probYesParent*math.log(probYesParent,2)+probNoParent*math.log(probNoParent,2))
    informationGains = []
    heads = list(x.columns)
    
    for i in range (1,len(heads)) :
        z = x[heads[i]].tolist()
        yesRCount = 0
        yesDCount = 0
        noRCount = 0
        noDCount = 0
        for j in range (len(z)):
            if z[j] == 'y' and tmp[j] == 'republican' :
                yesRCount +=1
            elif z[j] == 'y' and tmp[j] == 'democrat' :
                yesDCount +=1
            elif z[j] == 'n' and tmp[j] == 'republican' :
                noRCount +=1
            elif z[j] == 'n' and tmp[j] == 'democrat' :
                noDCount +=1
        if (noDCount + noRCount) == 0 or (yesDCount + yesRCount) == 0 :
            informationGain = 0
        else :
            yesDemocrat = yesDCount / (yesDCount + yesRCount)
            yesRepublic = yesRCount / (yesDCount + yesRCount)
            if (yesDemocrat == 0 and yesRepublic !=0) or (yesDemocrat != 0 and yesRepublic == 0):
                entropyYes = 0
            else :
                entropyYes = -(yesDemocrat*math.log(yesDemocrat,2) + yesRepublic*math.log(yesRepublic,2))
            
            noDemocrat = noDCount / (noDCount + noRCount) 
            noRepublic = noRCount / (noDCount + noRCount)
            if (noDemocrat == 0 and noRepublic !=0) or (noDemocrat != 0 and noRepublic == 0):
                entropyNo = 0
            else :
                entropyNo =  -(noDemocrat*math.log(noDemocrat,2) + noRepublic*math.log(noRepublic,2))
            informationGain = entropyParent - ((((yesDCount + yesRCount)/(y+n))*entropyYes) + (((noDCount + noRCount) / (y+n))*entropyNo))
        informationGains.append(informationGain)
        
    
    return informationGains.index(max(informationGains))+1
    


class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
    



class Tree:
    
    
    def __init__(self):
        self.root = None
   
    
    
    def updateTreeSize(self,node):
        if node == None :
            return 0
        
        return self.updateTreeSize(node.right) + self.updateTreeSize(node.left)+1
        
    
    def traverse(self,node):
        if node == None :
            return
        self.traverse(node.right)
        self.traverse(node.left)
        
        print(node.data)
        
                        
                    
def bulidBinarySearchTree(node,x,he,c = ""):
    
    if (isPureFeature(x, "republican") == True):
        node.data= "republican"
        return
    elif (isPureFeature(x, "democrat")== True):
        node.data= "democrat"
        return
    if(isPureFeature(x, "republican") == True):
        node.data= "republican"
        return
    elif (isPureFeature(x, "democrat")== True):
        node.data= "democrat"
        return
    
    
    if len(x.columns) == 2 :
        c = getMaxCountOfTarget(x)
    
    elif len(x.columns) == 1 :
        node.data = c
        return
   
    
    index = getMaxInformationGain(x)
    
    node.data = he[index]
   
    
    rightDataFrame,h = getDataFrameForAttribute(x,'y',he[index])
    node.right = Node("")
    bulidBinarySearchTree(node.right,rightDataFrame,h,c)
    
        
    leftDataFrame,h = getDataFrameForAttribute(x,'n',he[index])
    node.left = Node("")
    bulidBinarySearchTree(node.left,leftDataFrame,h,c)
    
    
def getMaxCountOfTarget(x) :
    goal = list(x['goals'])
    a = goal.count('republican')
    b = goal.count('democrat')
    if a > b:
        return 'republican'
    else :
        return 'democrat'
  

def correctData(x): 
    for i in headers :
        tmp = x[i].tolist()
        y = tmp.count('y')
        n = tmp.count('n')
        for j in range (len(tmp)):
            if tmp[j] == '?' and y > n :
                tmp[j] = 'y'
            elif tmp[j] == '?' and y<=n :
                tmp[j] = 'n'
        x[i] = tmp     
    return x


def splitData(ds,ratio):
    shuffle_df = ds.sample(frac=1)
    train_size = int((ratio/100) * len(ds))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    pd.DataFrame(train_set,columns = headers)
    pd.DataFrame(test_set,columns = headers)
    return train_set,test_set

def getTreePredictionForRow(x,root,headers) :
    curr = root  
    while curr.data != 'republican' and curr.data != 'democrat' :
        i = headers.index(curr.data)
        if x[i] == 'y' :
            curr = curr.right
        elif x[i] == 'n':
            curr = curr.left
    return curr.data
    

def calcAccuracy(x,root):
    right = 0
    wrong = 0
    
    h = list(x.columns)
    tmp = np.array(x)
   
    for i in range (x.shape[0]) : 
        if getTreePredictionForRow(tmp[i],root,h) == tmp[i][0]:
            right +=1
        else:
            wrong+=1
   
    calcAccuracy = right/x.shape[0]*100
    return calcAccuracy  

    


x = pd.read_csv("house-votes-84.data.txt",sep=",",names = headers)
x = correctData(x)

accuracies = []
treeSizes = []
print("First 5 runs with Training set size = 25 : ")    
for i in range(5) :
    xTrain,xTest = splitData(x,25)
    
    binaryTree = Tree()
    binaryTree.root = Node("")
    bulidBinarySearchTree(binaryTree.root,xTrain,headers)
    
    
    z = binaryTree.updateTreeSize(binaryTree.root)
    
    print("tree size is : ",z,"\n")
    treeSizes.append(z)
    n  = calcAccuracy(xTest,binaryTree.root)
    print("accuracy is : ",n,"\n")
    accuracies.append(n)
df = pd.DataFrame()
df['accuracies'] = accuracies
df['treeSizes'] = treeSizes
df.to_excel('first 5 Runs results.xlsx', index = False)  

runs = [30,40,50,60,70]

print("-----------------------------------------------------------")
df = pd.DataFrame()
maximumAcc = []
minimumAcc = []
averageAcc = []
maxTreeSize = []
minTreeSize = []
avgTreeSize = []
for i in runs :
    accuracies = []
    treeSizes = []
     
    for j in range(5) :
        xTrain,xTest = splitData(x,i)
        
        binaryTree = Tree()
        binaryTree.root = Node("")
        bulidBinarySearchTree(binaryTree.root,xTrain,headers)
        
        
        z = binaryTree.updateTreeSize(binaryTree.root)
          
        
        treeSizes.append(z)
        n  = calcAccuracy(xTest,binaryTree.root)
        accuracies.append(n)
    print("Training set Size = ",i)    
    maxAcc = max(accuracies)
    print("Maximum accuracy is : " , maxAcc)
    maximumAcc.append(maxAcc)
    minAcc = min(accuracies)
    print("minimum accuracy is : " , minAcc)
    minimumAcc.append(minAcc)
    avgAcc = sc.mean(accuracies)
    print("mean accuracy is : ", avgAcc)
    averageAcc.append(avgAcc)
    mxTS = max(treeSizes)
    print("Maximum treeSizes is : " , mxTS)
    maxTreeSize.append(mxTS)
    mnTS = min(treeSizes)
    print("minimum treeSizes is : " , mnTS)
    minTreeSize.append(mnTS)
    avTS = sc.mean(treeSizes)
    print("mean treeSizes is : ", avTS)
    avgTreeSize.append(avTS)
    print("------------------------------------------------------------")

df["Traning set Size"] = runs
df["Maximum accuracy"] = maximumAcc
df["Manimum accuracy"] = minimumAcc
df["mean accuracy"] = averageAcc
df["maximum Tree Size"] = maxTreeSize
df["minimum Tree Size"] = minTreeSize
df["mean Tree Size"] = avgTreeSize
    
    
df.to_excel('runs Report.xlsx', index = False)
   
    
    

