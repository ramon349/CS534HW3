from sklearn import datasets  
from hw3 import decisionTree
import pandas as pd 
import cProfile
import numpy as np  
from sklearn import metrics 
from sklearn.model_selection import RepeatedKFold

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
#binner(x,k=10)
predictions = np.zeros((x.shape[0],1) ) 
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
auc_list = list()
for train_index, test_index in rkf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index] 
    model = decisionTree(depth=2) 
    model.fit(X_train,y_train,num_sample=20)  
    predictions = np.zeros( (X_test.shape[0],1)) 
    for i in range(X_test.shape[0]):
        predictions[i] = model.predict(X_test[i]) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1) 
    auc = metrics.auc(fpr,tpr)
    print(f"Current AUC: {auc} --------")
    auc_list.append(metrics.auc(fpr, tpr)) 
print(np.mean(auc_list) )