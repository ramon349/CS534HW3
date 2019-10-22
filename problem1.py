from sklearn import datasets  
from hw3 import QuarternaryDecisionTree
import pandas as pd 
import cProfile
import numpy as np  
from sklearn import metrics 
from sklearn.model_selection import  KFold
from sklearn.tree import DecisionTreeClassifier 

""""NOTE: FOR TA
I had certain outputs written to txt file due to  runtime warnings preventing stdout from being readable.
I made a "unique" enough sounding text file not to overide anything in your systems by accident
""" 
f = open("log2_depth.txt",'w') 
#load data 
data = datasets.load_breast_cancer()
(x,y) = (data.data,data.target)  
predictions = np.zeros((x.shape[0],1) ) 
auc_list = list() 
# Train 2 models on the same dataset then test on the same dataset  
model = QuarternaryDecisionTree(depth=1) 
model.fit(x,y,num_sample=10)
for i in range(x.shape[0]):
    predictions[i] = model.predict(x[i]) 
fpr, tpr, thresholds = metrics.roc_curve(y, predictions, pos_label=1) 
my_auc = metrics.auc(fpr,tpr)
goal_mdl = DecisionTreeClassifier(criterion="entropy",max_depth=1)
goal_mdl.fit(x,y)
mdl_pred = goal_mdl.predict_proba(x)
fpr, tpr, thresholds = metrics.roc_curve(y,mdl_pred[:,1], pos_label=1) 
mdl_auc = metrics.auc(fpr,tpr)
f.write(f"my auc: {my_auc} VS .  model auc{mdl_auc} \n ")  
# Let's do 2 fold crossvaldiaiton 
rkf = KFold(n_splits=2)
for train_index, test_index in rkf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index] 
    model = QuarternaryDecisionTree(depth=2) 
    tst_model = DecisionTreeClassifier(criterion="entropy",max_depth=2)
    model.fit(X_train,y_train,num_sample=10)  
    tst_model.fit(X_train,y_train)
    predictions = np.zeros( (X_test.shape[0],1)) 
    for i in range(X_test.shape[0]):
        predictions[i] = model.predict(X_test[i])
    test_pred = tst_model.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1) 
    my_auc = metrics.auc(fpr,tpr) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_pred[:,1], pos_label=1) 
    test_mdl_auc = metrics.auc(fpr,tpr)
    f.write(f"my model AUC: {my_auc}  VS sklearn AUC:{test_mdl_auc} -------- \n") 
f.close() 
"""  Observations 
1) The quaterney decision tree outperforms the simple Decision tree for depths of 1 and 2 
2)Looking at the  AUC in cross valdiation we notice that the AUC of sklearns decision tree changes dramatically 
while the quaterny decision tree doesn't vary as much. This is a rather interesting observation as the 
quaterny decision tree has more complicated comparisons. It may also be that the complicated  comparisons help identify 
patterns that better represent the data when compared to a single threshold .
"""