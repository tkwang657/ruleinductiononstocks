# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:24:41 2023

@author: tszwu
"""

import numpy as np
import pandas as pd
def pickfeatures(sorteddict, num=10, threshold=0.5):
    rtn={}
    for j in sorteddict:
        toplist=list(sorteddict[j].keys())[:num]
        topfeats={}
        for i in toplist:
            if(abs(sorteddict[j][i])>=threshold):
                topfeats[i]=sorteddict[j][i]
        rtn[j]=topfeats
    return rtn



def normalisedata(table, norm='zscore', skip=['IsDivStock']):    
    rows=table.index.values
    cols=table.columns
    rtn=pd.DataFrame(index=rows, columns=cols)
    if(norm=='zscore'):
        for column in cols:
            if column in skip:
                rtn[column]=table[column]
                continue
            else:
                try:
                    mean=np.nanmean(table[column])
                    sd=np.nanstd(table[column])
                    if(np.isinf(mean))==True:
                        for r in rows:
                            if np.isinf(table[column][r])==True:
                                table.at[r, column]=np.nan
                            else:
                                pass
                    try:
                        mean=np.nanmean(table[column])
                        sd=np.nanstd(table[column])
                    except:
                        continue
                except Exception:
                    continue
                for r in rows:
                    if(sd!=0):
                        tmp=(table[column][r] - mean)/sd
                        rtn.at[r, column]=tmp
                    else:
                        rtn.at[r, column]=0

    if(norm=='minmax'):
        for column in cols:
            if(column in skip):
                rtn[column]=table[column]
                continue
            else:
                low = table[column][rows[0]]
                high = table[column][rows[0]]
                # print(column)
                for i in rows:
                    try:
                        flag = table[column][i] < low
                        if (flag):
                            low = table[column][i]
                    except Exception:
                        low = low
                    try:
                        flag = table[column][i] > high
                        if (flag):
                            high = table[column][i]
                    except Exception:
                        high = high
                for i in rows:
                    try:
                        rtn.at[i, column] = (table[column][i] - low) / (high - low)
                    # Non-numerical data is wiped, except for sector & dividends_all arr.
                    except Exception:
                        if column == "companySnapshot_sectorInfo" or "dividends_all":
                            rtn.at[i, column] = table[column][i]
                        else:
                            rtn.at[i, column] = np.nan

    return rtn

def z_score_intervals(z_scores, x):
    mean = 0
    std=1
    intervals = set()
    
    for i in range(-x, x+1):
        lower = mean + i * std
        upper = mean + (i+1) * std
        interval = (round(lower, 2), round(upper, 2))
        intervals.add(interval)

    return intervals

def discretisation(dataset, num=10): #pass in list of lists with lengths for each category
    # zscore<-1.65 --> p<0.1
    # zscore <-1.96 --> p<0.05
    # zscore <-2.58 --> p<0.01
    rows=dataset.index.values
    cols=dataset.columns
    rtn=pd.DataFrame(index=rows, columns=cols)
        
    for column in cols:
        #print(column)
        low=np.nanmin(dataset[column])
        high=np.nanmax(dataset[column])
        low, high =-max(abs(low), abs(high)), max(abs(low), abs(high))
        width=(high-low)/num
        intervals=[]
        for j in range(0, num):
            intervals.append([low+width*j, low+width*(j+1)])
        #print(column + ' intervals: ', intervals)
        for r in rows:
            #print(r)
            z=dataset[column][r]
            tmp=0
            for i in range(len(intervals)):
                if z==high:
                    tmp=num
                elif(z<intervals[i][1]) and (z>=intervals[i][0]):
                    tmp=i
                    break
                else:
                    if(np.isnan(z)):
                        tmp=np.nan
                    else:
                        tmp=2
            rtn.at[r, column]=tmp
    
    return rtn

def crosscorr(table, xcol, ycol=False, method='pearson'): #Calculate pairwise correlation between features given in x and y. x is a list of columns, same as y
    #method='pearson', 'kendall', 'spearman'
    if ycol:
        pass
    else:
        ycol=xcol
    CorrDict={}   #Sorted dictionary of correlations
    CorrMatrix=pd.DataFrame(index=xcol, columns=ycol)   #pandas dataframe of correlation
    for output in ycol:
        tmp={}
        try:
            y=pd.to_numeric(table[output])
        except:
            continue
        for column in xcol:
            try:
                x=pd.to_numeric(table[column])
            except:
                continue
            try:
                corr=x.corr(y, method=method)
            except:
                continue
            if(np.isnan(corr)):
                pass
            else:
                tmp[column]=corr
            CorrMatrix.at[column, output]=corr
        SortedCorr={}
        sortedx=sorted(tmp,reverse=True, key=lambda dict_key: abs(tmp[dict_key]))
        for key in sortedx:
            SortedCorr[key]=tmp[key]
        CorrDict[output]=SortedCorr
    
    return CorrMatrix, CorrDict

def linearfeatures(corrmatrix):
    #Prune high correlation features
    xcol=list(corrmatrix.columns)
    length=len(xcol)
    linearfeatures={}

    for i in range(length):
        tmp=[]
        tmp.append(xcol[i])
        for j in range(i+1, length):
            corrdiff=abs(corrmatrix[xcol[i]]-corrmatrix[xcol[j]])
            corrdiff=np.nanmean(corrdiff)
            if corrdiff<0.11 and corrmatrix[xcol[i]][xcol[j]]>= 0.95:
                tmp.append(xcol[j])
                linearfeatures[xcol[i]]=tmp
    tmp=[]
    xcol=list(linearfeatures.keys())
    for item in xcol:
        val=[j for j in linearfeatures[item]]
        tmp.append(val)

    i = 0
    #if two sets intersect, we want to take the union. If they don't, we leave them separately.
    linearset=[set(j) for j in tmp]
    while i < len(linearset):
        j = i + 1
        while j < len(linearset):
            if linearset[i].intersection(linearset[j]):
                linearset[i] = linearset[i].union(linearset[j])
                linearset.pop(j)
            else:
                j += 1
        i += 1
    return linearset
        # print(xcol[i],': ', tmp)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def ReduceFeatures(data, xcol):
    corrmatrix, sortedcorr=crosscorr(data, xcol=xcol, method='spearman')
    corrmatrix=corrmatrix.dropna(axis=1, how='all')
    corrmatrix=corrmatrix.dropna(axis=0, how='all')
    # print('Current x-features:' , xcol)
    linearfts=linearfeatures(corrmatrix)
    #Turn back to list
    linearfts=[[j for j in subset] for subset in linearfts]
    companies=data.index.values
    for j in range(len(linearfts)):
        for row in companies:
            try:
                columnnew=[data[linearfts[j][i]][row] for i in range(0, len(linearfts[j]))]
                newval=np.nanmean(columnnew)
            except:
                for i in range(len(columnnew)):
                    try:
                        flag=isfloat(columnnew[i])
                        if flag:
                            pass
                        else:
                            columnnew[i]=np.nan
                    except:
                        columnnew[i]=np.nan
                        
            data.at[row, linearfts[j][0]]=newval
        data=data.drop([linearfts[j][i] for i in range(1, len(linearfts[j]))], axis=1)
    return data

