from sklearn import datasets  
from hw3 import DaRDecisionTree
import cProfile
import numpy as np  
from sklearn.metrics import  accuracy_score ,mean_squared_error
from sklearn.linear_model import Ridge 
import matplotlib.pyplot as plt 

data = datasets.load_boston()

(x,y) = (data.data,data.target)  
model = DaRDecisionTree(depth=2) 
model.fit(x,y) 
predictions = np.zeros((x.shape[0],)) 

for i in range(x.shape[0]):
    predictions[i] = model.predict(x[i,:].reshape((1,13)) ) 
my_model= mean_squared_error(y,predictions) 
other_model =  Ridge()
other_model.fit(x,y) 
predi = other_model.predict(x) 
other_model_perf = mean_squared_error(y,predi) 


print(f"My model: {my_model} MSE . vs  other model: {other_model_perf}") 
plt.plot(predictions)  
plt.plot(y)
plt.show()