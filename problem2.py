from sklearn import datasets  
from hw3 import DaRDecisionTree
import cProfile
import numpy as np  
from sklearn.metrics import  accuracy_score ,mean_squared_error
from sklearn.linear_model import Ridge 
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedKFold

data = datasets.load_boston()
#loading the data 
(x,y) = (data.data,data.target) 
for e in range(0,4):
    model = DaRDecisionTree(depth=e) 
    model.fit(x,y) 
    other_model = Ridge()
    other_model.fit(x,y)
    predictions = np.zeros((x.shape[0],)) 
    for i in range(x.shape[0]):
        predictions[i] = model.predict(x[i].reshape(1,-1) )  
    my_model= mean_squared_error(y,predictions) 
    predi = other_model.predict(x) 
    other_model_perf = mean_squared_error(y,predi) 
    print(f" With depth {e} tree model MSE is: {my_model} Compared to standard ridge mse: {other_model_perf}") 
splitter = RepeatedKFold(n_splits=2,n_repeats=15)  
for d in range(0,4):
    my_mse = list() 
    standard_mse = list()
    for test_index,train_index in splitter.split(x): 
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]  
        model = DaRDecisionTree(depth=d) 
        model.fit(X_train,y_train) 
        other_model = Ridge()
        other_model.fit(X_train,y_train)
        predictions = np.zeros((X_test.shape[0],)) 
        for i in range(X_test.shape[0]):
            predictions[i] = model.predict(X_test[i].reshape(1,-1) ) 
        my_model_mse= mean_squared_error(y_test,predictions)  
        my_mse.append(my_model_mse)
        predi = other_model.predict(X_test)  
        other_model_mse = mean_squared_error(y_test,predi) 
        standard_mse.append(other_model_mse)
    print(f"For depth {d}  my model MSE: {np.mean(my_mse)} standard ridge: {np.mean(standard_mse) } ") 
""" Observations on performance 
1) When training and testing on the entirety  of the dataset we notice a decrease in MSE as we increase the depth. 
2) Validating performance using 10 repeats of 3 fold crossvalidation shows that our modells performance varies when 
exposed to new data. The worse average performance may be do to increased variance of our model. On inspection there are 
several cases where the tree does indeed outperform the ridge regression model. 
"""