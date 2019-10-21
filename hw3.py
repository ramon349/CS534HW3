from sklearn.linear_model import Ridge
from itertools import combinations ,product
import numpy as np 
import math
import scipy.stats as sstats
import time 
class decisionTree: 
    def __init__(self,cost=None,depth=2):
        self.decision = 0 
        self.depth =depth
        self.children = None 
        self.num_sample = 0 
    def fit(self,x,y,num_sample=None):  
        if self.depth ==0:  
            #handle case where we've reached the bottom 
            self.decision = np.mean(y) #sstats.mode(y).mode
            self.children = None 
        else: 
            self.num_sample =num_sample
            self.decision = None 
            self.children =list() 
            #case where we are not the bottom and need to split 
            (cost,f1,f2,f1_val,f2_val) = self.get_best(x,y) 
            print(f"With cost {cost} we found f1: {f1} t: {f1_val} and f2:{f2} t:{f2_val}")
            #set up decision variables 
            self.feat_1 = f1 
            self.feat_1_thresh = f1_val 
            self.feat_2 = f2 
            self.feat_2_thresh = f2_val 
            l_f1 = x[:,f1] <= f1_val 
            l_f2 = x[:,f2] <= f2_val 
            g_f1 =  np.logical_not(l_f1)
            g_f2 = np.logical_not(l_f2) 
            buckets = [ np.logical_and(l_f1,l_f2),np.logical_and(l_f1,g_f2),np.logical_and(g_f1,l_f2),np.logical_and(g_f1,g_f2)]
            for b in buckets: 
                mdl = decisionTree(depth =self.depth-1)  
                mdl.fit(x[b],y[b],num_sample=self.num_sample)
                self.children.append(mdl)
    def predict(self,x):
        if self.decision != None:
            return self.decision 
        else: 
            l_f1 = x[self.feat_1] <= self.feat_1_thresh
            l_f2 = x[self.feat_2] <= self.feat_2_thresh
            g_f1 =  np.logical_not(l_f1)
            g_f2 = np.logical_not(l_f2) 
            buckets = [ np.logical_and(l_f1,l_f2),np.logical_and(l_f1,g_f2),np.logical_and(g_f1,l_f2),np.logical_and(g_f1,g_f2)] 
            for i,b in enumerate(buckets): 
                if b:
                    return  self.children[i].predict(x)
    def generate_unique_values(self,x):
        """ Instead of computing thresholds each time let's pre compute them
        by creating a list of unique values for each feature.
        x: our feature matrix 
        """
        unique_list = list() 
        for i in range( x.shape[1]):
            unique_list.append( self.inbetween_vals(x[:,i]))  
        return unique_list
    def get_best(self,X,Y):  
        combi = combinations(range(X.shape[1]),2) 
        best_tuple = (np.inf,0,0) 
        best_pair = (0,0)
        unique_list = self.generate_unique_values(X)
        for f1,f2 in combi:  
            current_cost  = self.eval_metrix(X[:,[f1,f2]],Y,unique_list[f1],unique_list[f2])  #shape of currenct_cos (c_cost,f1_t,f2_t) 
            if current_cost[0] < best_tuple[0]:
                best_tuple = current_cost 
                best_pair = (f1,f2) 
        return  (best_tuple[0],best_pair[0],best_pair[1],best_tuple[1],best_tuple[2])
    def eval_metrix(self,x,y,f1_unique,f2_unique): 
        feat_pairs = list(product(f1_unique,f2_unique)  ) #let's look at every combination  of these feature values    
        best_tuple = (np.inf,0,0)
        for f1,f2 in feat_pairs: 
            current_cost = self.eval_cost(x,y,f1,f2) # shape of current loss (LL,f1_val,f2_val) 
            if current_cost[0] < best_tuple[0] :
                best_tuple = current_cost  
        return best_tuple
    def inbetween_vals(self,whole_x): 
        """ inbetween_vals generates thresholds based on middle point between unique values  
        whole_x is entire column vector of feature values 
        y is the albel 
        """  
        u_vals  = np.unique(whole_x) 
        if self.num_sample: 
            u_vals  = np.unique(whole_x) 
            potential_vals = np.zeros((self.num_sample,1))
            min_u = min(u_vals) 
            counter = 0
            for i in range(0,len(u_vals),math.ceil(len(u_vals)/self.num_sample) ): 
                potential_vals[counter] =  u_vals[i] 
                counter +=1 
        else: 
            potential_vals = u_vals 
        inbetween = list()   
        for i in range(len(potential_vals) -1): 
            inbetween.append( (potential_vals[i] + potential_vals[i+1] )/2)   
        return inbetween

    def eval_cost(self,x,y,f1_val,f2_val):
        """ since this is a 4-way decision problem. we can no longer asign onse side is negativ eone side is positive. 
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
            return (math.inf,f1_val,f2_val) 
        else: 
            return (LL,f1_val,f2_val)
    def get_bucket_performance(self,real_labels):  
        if real_labels.size != 0:  
            p = np.mean(real_labels)
            out1 = -1*p*np.log(p) 
            out2 = -1*(1-p)*np.log(1-p) 
            if out1 == -1*math.inf: 
                out1 = 0  
            if out2 == -1*math.inf: 
                out2 = 0 
            return real_labels.size*(out1 + out2)
        else:
            return 0
class DaRDecisionTree(decisionTree):
    def __init__(self,depth=2):  
        self.depth = depth 
        self.decision =0 
        self.children=None
    def fit(self,x,y):
        t = time.time() 
        if self.depth ==1:  
            #handle case where we've reached the bottom 
            self.decision =  Ridge()
            self.decision.fit(x,y)
            self.children = None 
        else:  
            self.children = list()
            self.decision = None 
            #case where we are not the bottom and need to split 
            (cost,f1,f1_val) = self.get_best(x,y) 
            print(f"With cost {cost} we found f1: {f1} t: {f1_val}")
            #set up decision variables 
            self.feat_1 = f1 
            self.feat_1_thresh = f1_val 
            l_f1 = x[:,f1] < f1_val 
            g_f1 =  np.logical_not(l_f1) 
            buckets = [l_f1,g_f1] 
            for b in buckets:  
                mdl = DaRDecisionTree( depth = self.depth -1 ) 
                mdl.fit(x[b],y[b] ) 
                self.children.append(mdl) 
    def predict(self,x):
        if self.decision != None:
            return self.decision.predict(x)
        else: 
            l_f1 = x[0,self.feat_1] <= self.feat_1_thresh
            g_f1 =  np.logical_not(l_f1)
            buckets = [ l_f1, g_f1 ]  
            for i,b in enumerate(buckets): 
                if b:
                    return  self.children[i].predict(x) 

    
    def get_best(self,X,Y):  
        """ in this case were only splititng on a single variable 
            only iterate through a single list 

        """
        t = time.time()
        best_tuple = (np.inf,0) 
        best_feat = 0
        unique_list = self.generate_unique_values(X)
        for f1 in range(X.shape[1] ): #iterate through each field 
            current_cost  = self.eval_metrix(X[:,f1],Y,unique_list[f1])  #shape of currenct_cos (c_cost,f1_t) 
            if current_cost[0] < best_tuple[0]:
                best_tuple = current_cost 
                best_feat = f1 
            #print(f"We finished evaluating one feature set out of{X.shape[1]} time: {time.time() - t}")
        return  (best_tuple[0], best_feat, best_tuple[1] )
    def eval_metrix(self,x,y,f1_unique): 
        t = time.time()
        best_tuple = (np.inf,0,0)
        for thresh_candidate in f1_unique :
            current_cost = self.eval_cost(x,y,thresh_candidate) # shape of current loss (LL,f1_val) 
            if current_cost[0] < best_tuple[0]:
                best_tuple = current_cost   
        #print(f"Finished splits for a feature cost{best_tuple[0]} time:{time.time() - t} ")
        return best_tuple
    def inbetween_vals(self,whole_x): 
        """ inbetween_vals generates thresholds based on middle point between unique values  
        whole_x is entire column vector of feature values 
        y is the albel 
        """ 
        potential_vals = np.unique(whole_x) 
        inbetween = list()   
        for i in range(len(potential_vals) -1): 
            inbetween.append( (potential_vals[i] + potential_vals[i+1] )/2)   
        return inbetween

    def eval_cost(self,x,y,f1_val):
        """ since this is a 4-way decision problem. we can no longer asign onse side is negativ eone side is positive. 
            we may wish to utilize the  most common class in each of the corresponding buckets as our prediciton label 
            this would lead us to the potential if lacking proper class splits producing a null classifier
        """
        l_f1 = x <= f1_val 
        g_f1 =  np.logical_not(l_f1) 
        buckets = [l_f1,g_f1]
        LL= 0
        for bucket in buckets:
            my_LL = self.get_bucket_performance(y[bucket]) 
            LL += my_LL 
        if LL == math.inf or math.isnan(LL): 
            return (1*math.inf,f1_val) 
        else: 
            return (LL,f1_val)
    def get_bucket_performance(self,real_labels):  
        if real_labels.size != 0: 
            return real_labels.size*np.var(real_labels)
        else:
            return 0


