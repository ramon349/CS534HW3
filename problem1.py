from sklearn import datasets  
from hw3 import decisionTree
import cProfile



data = datasets.load_breast_cancer()

(x,y) = (data.data,data.target)
model = decisionTree(x,y) 