from sklearn import datasets  
from hw3 import decisionTree
import pandas as pd 
import cProfile
import numpy as np  
from sklearn.metrics import  accuracy_score

def binner(x,k=10):
    new_x = np.zeros((x.shape[0]))
    for i in range(x.shape[1]): 
        min_x = np.min(x[:,i]) 
        max_x = np.max(x[:,i]) 
        step = (max_x-min_x)/k
        bins = np.arange(min_x,max_x,step=step)  
        print(f"lenght of bins {len(bins) }")
        print(f"{max_x}: {min_x}") 
        bin_val =1 
        for j in range(len(bins) -1): 
            group = np.logical_and(x[:,i] >= bins[j],x[:,i] < bins[j+1] ) 
            new_x[group] =  bin_val 
            print(np.sum(group))
            bin_val +=1 
        x[:,i] = np.copy(new_x)  

data = datasets.load_breast_cancer()
(x,y) = (data.data,data.target)  
binner(x,k=10)
model = decisionTree(depth=2) 
model.fit(x,y)
predictions = np.zeros((x.shape[0],1) ) 
for i in range(x.shape[0]):
    predictions[i] = model.predict(x[i,:]) 
w = accuracy_score(y,predictions)
print(w)
