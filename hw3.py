from itertools import combinations ,product
import numpy as np 

class decisionTree: 
    def __init__(self): 
        self.decision = 0 
        self.left = None 
        self.right = None 
        print("hello cruel world")
    def fit(self,X,y,cost='log',depth=5):
        """  Method to fit my decison tree 
            X:numpy array nx30 data array contianing training data. Should be continious variables 
            y: categorial  
            cost: looking at the cost funciton  used to make decisions 
            depth: maximum depth of tree 

            fit  should if not a decision node provide the best split then initalize 2 other decision tress and fit them using the substets 

        """  
        if depth == 0: 
            return  

        (f1,f2,f1_val,f2_val) = self.get_best(X,y)

    def get_best(self,X,Y): 
        combi = combinations(range(X.shape[1]),2)
        for f1,f2 in combi: 
            print(f"we got  f1: {f1} and f2: {f2}")  
        return (1,2,3,4) 
    def eval_metrix(x,y): 
        f1_uniques = np.unique(x[:,0])
        f2_unique = np.unique(x[:,1])
        for f1,f2 in product(f1_unique,f2_unique): 
            print("pokemon")
    def eval_cost(x,y,f1_val,f2_val):
       """since this is a way decision problem. we can no longer asign onse side is negativ eone side is positive. 
            we may wish to utilize the  most common class in each of the corresponding buckets as our prediciton label 
            this would lead us to the potential if lacking proper class splits producing a null classifier
       """
        l_f1 = x[:,0] <= f1_val 
        l_f2 = x[:,1] <= f2_val 
        g_f1 =  np.logical_not(l_f1)
        g_f2 = np.logical_not(x[:,1] > f2_val )
        q_1 = np.sum(l1_f1)
        q_2 = np.sum(l_f2) 
        p_1 = x.shape[0] - q_1
        p_2 = x.shape[0] - q_2 
        LLs = np.zeros((1,4))
        LLs[0] = get_bucket_performance(np.logcial_and(l_f1,l_f2),)
        # we may instead want to get the performance of individual buckets 
    def get_bucket_performance(pred_labels,real_labels,q,p):
        #this would be performnace for a single child node  so it would bea column of the decision table 
        #you would need a filtrer opteratio nusign the logicla labels to get only the labels of intetest
        # calcualte true positives, true negatives and so forth 

