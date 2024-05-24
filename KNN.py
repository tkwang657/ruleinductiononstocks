# -*- coding: utf-8 -*-
"""
Created on Mon May  8 01:29:17 2023

@author: tszwuLEN
"""
import pandas as pd
import numpy as np

def distance(x1, x2, dist_metric='euclidean', p=2): #Takes in vectors as a pandas dataframe of numerical data or Nans
    missing1=np.isnan(x1)
    missing2=np.isnan(x2)
    missing=missing1+missing2 
    vec1=np.array(x1)[~missing]
    vec2=np.array(x2)[~missing]
    scale=len(x1)/len(vec1)
    match dist_metric:
        case 'euclidean':                    
            rtn=np.power(scale * np.sum(np.power(np.abs(np.subtract(vec1 , vec2)), 2)), 1/2)
        case 'minkowski':
            rtn=np.power(scale * np.sum(np.power(np.abs(np.subtract(vec1 , vec2)), p)), 1/p)
        case 'cosine':
            rtn=np.dot(vec1, vec2)/ (np.sqrt((np.sum(np.square(vec1)))) * np.sqrt((np.sum(np.square(vec2)))))
    
    return rtn
            
    
    
def cdist(table, dist_metric='euclidean', cols=False, p=2): #dist is a distance function
    indices=table.index.values
    rtn=pd.DataFrame(index=indices, columns=indices)
    if(cols):
        pass
    else:
        cols=table.columns
    length=len(indices)
    for i in range(0, length):
        stock1=indices[i]
        rtn.at[stock1, stock1]=0
        for j in range(i+1, len(indices)):
            stock2=indices[j]   
            vec1=[table[col][stock1] for col in cols]
            vec2=[table[col][stock2] for col in cols]
            d=distance(vec1, vec2, dist_metric=dist_metric, p=p)
            rtn.at[stock1, stock2]=d
            rtn.at[stock2, stock1]=d
    return rtn
dist_metric = 'euclidean'
k = 5

def kNearest(table,  xcol,k=11, dist_metric='euclidean', p=2 ):
    xvar=table[xcol]
    indices=table.index.values
    distances = cdist(xvar, cols=xcol, dist_metric=dist_metric, p=p)
    knn_indices = np.argsort(distances)
    print(type(knn_indices))
    neighbors = knn_indices[:, 1:k]
    Knearest={}
    print(distances)
    for i in range(len(indices)):        
        Knearest[indices[i]]=[(j, d) for j, d in zip(indices[neighbors[i]].tolist(), distances.loc[indices[i]][indices[neighbors[i]]].tolist())]
    return Knearest