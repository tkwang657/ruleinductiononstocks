import numpy as np
import pandas as pd



def elementaryset(data, attributes=False):
    if (attributes==False):
        attributes=data.columns
    else:
        pass
    items=data.index.values
    B = []
    discarded=[]
    
    
    #Drop all nan columns
    
    
    for item in items:
        if item in discarded:
            continue
        else:
            row1=data.loc[item][attributes]
            selected_index = []
            for item2 in items:
                row2=data.loc[item2][attributes]
                if list(row1) == list(row2):
                    selected_index.extend([item2])
                    discarded.append(item2)
                B.append(selected_index)

	# remove the duplicates
    B = [x for n, x in enumerate(B) if x not in B[:n]]
    print('B* = ',B)
    if(len(B)==len(data)):
        print('Trivial Elementary Set')
    return B

def flatten(lst):
    # """
    # Recursively flattens a collection of lists.
    # """
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def consistent(B, D):   #checks if a smaller elementary set B is consistent with a larger elementary set D
    for b in B:
        flag=0
        for d in D:
            if set(b) <= set(d):
                flag=1
                continue
        if(flag==0):
            # print('Inconsistent subset of items:\n', b)
            return False
    return True

def smallestcover(data, D, attributes): 
    # """
    # Function: Given a global, consistent and covering set of attributes (consistent with elementary set of decisionis D), find the smallest covering subset S.
    #
    #
    #
    # Parameters
    # ----------
    # data : TYPE
    #     DESCRIPTION.
    # D : List of Lists 
    #     Elementary set of decisions.
    # attributes : List of covering set of attributes
       

    # Returns
    # -------
    # Smallest consistent and covering subset S of attributes
    # """
    
    if attributes:
        pass
    else:
        print("Please pass a covering list of attributes")
        return False
    subcover=[i for i in attributes]
    flag=True
    while flag:
        discard=''
        consistency=True
        print("Current Cover: ", subcover)
        for col in subcover:
            print('Testing: ', col)
            test=[i for i in subcover if i!=col]
            B_small=elementaryset(data, attributes=test)
            if consistent(B_small, D):
                consistency=True
                discard=col
                flag=True
                break
            else:
                consistency=False
                flag = False
        if(consistency):
            print('Removing: ', discard, '\n')
            subcover.remove(discard)
    #sanity check:
    B_new=elementaryset(data, subcover)
    if consistent(B=B_new, D=D):
        print('Minimal Subcover: ', subcover)
        return subcover
    else:
        return False
            
def inferrule(data, subcover, decisions):    
    # """

    # Parameters
    # ----------
    # data : type=pd.DataFrame
    # subcover : Minimal Covering set of features for decision outputs
    #     type=List of features
    
    # Returns
    # -------
    # Condition for a set of rules
    # """
    index=data.index.values
    RuleSet={}
    B=elementaryset(data, subcover) #Collection of sets of stocks that agree on all features in the subcover
    #Create Covering, consistent Ruleset
    
    for x in B:
        rowx=data.loc[x[0]]
        conditions=[]
        decision=[]
        dropping=True
        cover=[j for j in subcover]
        while dropping:
            for feature in cover:
                #try dropping a feature
                if(feature==cover[-1]):
                    dropping=False
                test=[i for i in cover if i!=feature]
                rowxtest=rowx[test]  #Case representation in reduced feature space
                satisfy=0
                for row in index:
                    if list(rowxtest)==list(data.loc[row][test]) and (row in x):
                        satisfy=1 
                    elif list(rowxtest)==list(data.loc[row][test]) and (row not in x):
                        #feature can't be dropped
                        satisfy=0
                        break
                if satisfy==1:
                    #No contradictions so feature can be dropped locally
                    print("Discarding: ", feature)
                    cover.remove(feature)
                    break
                else:
                    #can't drop this feature so go to next
                    continue
        
        localsubcover=cover
        for i in localsubcover:
            conditions.append((i,rowx[i]))
        print(conditions)
        if len(conditions)==1:
            conditions=tuple(conditions[0])
        else:
            conditions=tuple(conditions)
        for j in decisions:
            decision.append((j, rowx[j]))
        localrule={conditions:decision}
        RuleSet[conditions]=decision

    return RuleSet
    


def main(discretized_data, attributes, decisions):
    B=elementaryset(discretized_data, attributes=attributes)
    D=elementaryset(discretized_data, attributes=decisions)
    if consistent(B=B, D=D):
        print('Dataset is consistent')
    else:
        print("Dataset is not consistent")
        return False, False
    cover_min=smallestcover(discretized_data, D, attributes)
    if(cover_min==False):
        print("Error in finding minimal covering")
        return False, False
    elif len(cover_min)==len(attributes):
        print("No strictly smaller covering of features")
        return False, False
    else:
        Rules=inferrule(discretized_data, subcover=cover_min, decisions=decisions)
        return cover_min, Rules
    
    
        
                
        
    