from itertools import combinations ,product
import numpy as np 
import math
import scipy.stats as sstats
import time 
class decisionTree: 
    def __init__(self,x,y,cost=None,depth=5):  
        if depth ==0:  
            #handle case where we've reached the bottom 
            self.decision = sstats.mode(y) 
            self.left =None 
            self.right =None 
        #case where we are not the bottom and need to split 
        (cost,f1,f2,f1_val,f2_val) = self.get_best(x,y)
    def fit(self,X,y,cost='log',depth=5): 
        print("hi")

    def get_best(self,X,Y):  
        t = time.time() 
        combi = combinations(range(X.shape[1]),2) 
        my_list = list()  
        counter = 0 
        best_tuple = (-1*np.inf,0,0) 
        best_pair = (0,0)
        for f1,f2 in combi:  
            current_cost  = self.eval_metrix(X[:,[f1,f2]],Y)  #shape of currenct_cos (c_cost,f1_t,f2_t) 
            print(f"finished one combination after {time.time() -t}")
            if current_cost[0] > best_tuple[0]:
                best_tuple = current_cost 
                best_pair = (f1,f2)
        return  (best_tuple[0],best_pair[0],best_pair[1],best_tuple[0],best_tuple[1])
    def inspect_x(self,x,y): 
        print("---------comp-------------")
        for i in range(2):
            dist_1 = np.mean(x[y==1,i])
            dist_2 = np.mean(x[y!=1,i]) 
            print(f"hello the means are {dist_1} and {dist_2} ")  
    def eval_metrix(self,x,y): 
        t = time.time() 
        self.inspect_x(x,y)
        f1_unique = self.inbetween_vals(np.unique(x[:,0]) )
        f2_unique = self.inbetween_vals(np.unique(x[:,1]) ) 
        my_list = list() 
        feat_pairs = list(product(f1_unique,f2_unique)  ) #let's look at every combination  of these feature values    
        best_tuple = (-1*np.inf,0,0)
        print(f"Starting to evaluate feature pairs: {time.time() - t}")
        for f1,f2 in feat_pairs: 
            current_cost = self.eval_cost(x,y,f1,f2) # shape of current loss (LL,f1_val,f2_val) 
            if current_cost[0] > best_tuple[0] :
                best_tuple = current_cost  
        print(f"Finished evaluating feature pairs: {time.time() - t } ") 
        print(f"Total elements was {len(feat_pairs)}")
        return best_tuple
    def inbetween_vals(self,x): 
        inbetween = list()  
        for i in range(len(x)-1): 
            inbetween.append( (x[i] + x[i+1] )/2)
        return inbetween

    def eval_cost(self,x,y,f1_val,f2_val):
        """ since this is a way decision problem. we can no longer asign onse side is negativ eone side is positive. 
            we may wish to utilize the  most common class in each of the corresponding buckets as our prediciton label 
            this would lead us to the potential if lacking proper class splits producing a null classifier
        """
        l_f1 = x[:,0] <= f1_val 
        l_f2 = x[:,1] <= f2_val 
        g_f1 =  np.logical_not(l_f1)
        g_f2 = np.logical_not(l_f2) 
        buckets = [ np.logical_and(l_f1,l_f2),np.logical_and(l_f1,g_f2),np.logical_and(g_f1,l_f2),np.logical_and(g_f1,g_f2)]  
        LL= 0
        for bucket in buckets:
            my_LL = self.get_bucket_performance(y[bucket]) 
            LL += my_LL 
        if LL == math.inf or math.isnan(LL): 
            return (-1*math.inf,f1_val,f2_val) 
        else: 
            return (LL,f1_val,f2_val)
    def get_bucket_performance(self,real_labels):  
        if real_labels.size != 0: 
            common_class = sstats.mode(real_labels)  # self.get_common_label(real_labels) 
            correct = np.sum(real_labels == common_class)
            incorrect = real_labels.size - correct # instead of redoign the sum lets just get the difference np.sum(real_labels != common_class) 
            p = np.mean(real_labels)
            return correct*np.log(p)  + incorrect*np.log(1-p)
        else:  
            return 0 
        #need an if statement for being a negative node or a positive node 
        #this would be performnace for a single child node  so it would bea column of the decision table 
        #you would need a filtrer opteratio nusign the logicla labels to get only the labels of intetest
        # calcualte true positives, true negatives and so forth 
    def get_common_label(self,y):  
        u_labels = np.unique(y)
        class_counts = np.zeros((len(u_labels),2))  
        for i,label in enumerate(u_labels): 
            class_counts[i,0] = label 
            class_counts[i,1] = np.sum(y==label)
        class_counts.sort(axis=1)  
        return class_counts[-1,1]
