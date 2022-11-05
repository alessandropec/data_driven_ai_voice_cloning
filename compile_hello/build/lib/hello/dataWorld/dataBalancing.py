from imblearn.over_sampling import RandomOverSampler,SMOTE,SMOTENC,SVMSMOTE,ADASYN,BorderlineSMOTE,KMeansSMOTE

from imblearn.under_sampling import RandomUnderSampler,ClusterCentroids,CondensedNearestNeighbour,\
                                    EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,\
                                    InstanceHardnessThreshold,NearMiss,NeighbourhoodCleaningRule,\
                                    OneSidedSelection,TomekLinks

from imblearn.combine import SMOTEENN,SMOTETomek

from collections import Counter

import pandas as pd

COMBINED_SAMPLING_ALGORITHMS=["SMOTEENN","SMOTETomek"]

UNDER_SAMPLING_ALGORITHMS=[
                            "RandomUnderSampler","ClusterCentroids","CondensedNearestNeighbour",\
                            "EditedNearestNeighbours","RepeatedEditedNearestNeighbours","AllKNN",\
                            "InstanceHardnessThreshold","NearMiss","NeighbourhoodCleaningRule",\
                            "OneSidedSelection","TomekLinks"
                        ]
                        
OVER_SAMPLING_ALGORITHMS=["RandomOverSampler","SMOTE","SMOTENC","SVMSMOTE","ADASYN","BorderlineSMOTE","KMeansSMOTE"]

def over_sampling(df,target_col,algorithm="RandomOverSampler",random_state=None,n_jobs=None,verbose=False):
    model=None
    if algorithm=="RandomOverSampler":
        model=RandomOverSampler(random_state=random_state)
    elif algorithm=="SMOTE":
        model=SMOTE(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="SMOTENC":
        model=SMOTENC(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="ADASYN":
        model=ADASYN(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="BorderlineSMOTE":
        model=BorderlineSMOTE(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="KMeansSMOTE":
        model=KMeansSMOTE(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="SVMSMOTE":
        model=SVMSMOTE(random_state=random_state,n_jobs=n_jobs)
    
    else:
        print("Error, the algorithm specified not exists or is not avalaible. Check Imbalanced learn Over-sampling methods documentation.")
        return

    x_res,y_res=model.fit_resample(df[list(set(df.columns)-set(target_col))],df[target_col])

    df_re=pd.DataFrame(x_res,columns=list(set(df.columns)-set(target_col)))
    df_re[target_col]=y_res

    if verbose==True:
       print('Original dataset shape %s' % Counter(df[target_col]))
       print('Resampled dataset shape %s' % Counter(df_re[target_col]))

    return df_re


def under_sampling(df,target_col,algorithm="RandomUnderSampler",random_state=None,n_jobs=None,verbose=False):
    model=None
    if algorithm=="RandomUnderSampler":
        model=RandomUnderSampler(random_state=random_state)
    elif algorithm=="ClusterCentroids":
        model=ClusterCentroids(random_state=random_state)
    elif algorithm=="CondensedNearestNeighbour":
        model=CondensedNearestNeighbour(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="EditedNearestNeighbours":
        model=EditedNearestNeighbours(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="RepeatedEditedNearestNeighbours":
        model=RepeatedEditedNearestNeighbours(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="AllKNN":
        model=AllKNN(n_jobs=n_jobs)
    elif algorithm=="InstanceHardnessThreshold":
        model=InstanceHardnessThreshold(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="NearMiss":
        model=NearMiss(n_jobs=n_jobs)
    elif algorithm=="NeighbourhoodCleaningRule":
        model=NeighbourhoodCleaningRule(n_jobs=n_jobs)
    elif algorithm=="OneSidedSelection":
        model=OneSidedSelection(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="TomekLinks":
        model=TomekLinks(n_jobs=n_jobs)


    else:
        print("Error, the algorithm specified not exists or is not avalaible. Check Imbalanced learn Under-sampling methods documentation.")
        return
    x_res,y_res=model.fit_resample(df[list(set(df.columns)-set(target_col))],df[target_col])

    df_re=pd.DataFrame(x_res,columns=list(set(df.columns)-set(target_col)))
    df_re[target_col]=y_res

    if verbose==True:
       print('Original dataset shape %s' % Counter(df[target_col]))
       print('Resampled dataset shape %s' % Counter(df_re[target_col]))

    return df_re

def combine_sampling(df,target_col,algorithm="SMOTEEN",random_state=None,n_jobs=None,verbose=False):
    model=None
    if algorithm=="SMOTEEN":
        model=SMOTEENN(random_state=random_state,n_jobs=n_jobs)
    elif algorithm=="SMOTETomek":
        model=SMOTETomek(random_state=random_state,n_jobs=n_jobs)
    
    else:
        print("Error, the algorithm specified not exists or is not avalaible. Check Imbalanced learn combined-sampling methods documentation.")
        return

    x_res,y_res=model.fit_resample(df[list(set(df.columns)-set(target_col))],df[target_col])

    df_re=pd.DataFrame(x_res,columns=list(set(df.columns)-set(target_col)))
    df_re[target_col]=y_res

    if verbose==True:
       print('Original dataset shape %s' % Counter(df[target_col]))
       print('Resampled dataset shape %s' % Counter(df_re[target_col]))

    return df_re