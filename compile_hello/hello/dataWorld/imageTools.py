import numpy as np
import pandas as pd
from tqdm import tqdm

def rgb_to_flat(img):    
    x=np.array(img)
    x=x.reshape(x.shape[0]*x.shape[1]*x.shape[2])
    
    return x

def get_flat_df(imgDF):
    imgFlat=imgDF.apply(lambda v: rgb_to_flat(v))
    tmp=[]
    for img in tqdm(imgFlat):
        tmp.append(img)
    return pd.DataFrame(tmp)

def rgb_to_avg(img):    
    x=np.array(img)
    x=np.mean(x,axis=-1)
    x=x.reshape(x.shape[0]*x.shape[1])
    
    return x

def get_avg_df(imgDF):
    imgFlat=imgDF.apply(lambda v: rgb_to_avg(v))
    tmp=[]
    for img in tqdm(imgFlat):
        tmp.append(img)
    return pd.DataFrame(tmp)