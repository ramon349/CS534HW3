from sklearn import datasets  
from hw3 import decisionTree
import pandas as pd 
import cProfile
import numpy as np 

def binner(x,k=10):  
    for i in range(x.shape[1]): 
        w,b= pd.cut(x[:,i],bins=k,labels=False,retbins=True)  
        x[:,i] = w 
data = datasets.load_breast_cancer()
(x,y) = (data.data,data.target)  
binner(x,k=10)
model = decisionTree(x,y)  
breakpoint() 
print("Hi")