from sklearn import datasets  
from hw3 import decisionTree
import cProfile
import numpy as np 

def binner(x,k=10): 

data = datasets.load_breast_cancer()
(x,y) = (data.data,data.target) 
bi
#model = decisionTree(np.round(x),y) 