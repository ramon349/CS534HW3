from sklearn import datasets  
from hw3 import decisionTree
import cProfile
import numpy as np 



data = datasets.load_breast_cancer()

(x,y) = (data.data,data.target)
model = decisionTree(np.round(x),y) 