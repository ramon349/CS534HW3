from sklearn import datasets  
from hw3 import decisionTree




data = datasets.load_breast_cancer()

(x,y) = (data.data,data.target)
model = decisionTree()
model.fit(x,y)