##########################################
#Audio Dataset structure:
#   root
#       |+spID_spNameID_spSex_spAge_spRegion_spCity
#           |+spID_spNameID_spSex_spAge_spRegion_spCity_spID_trackID
#
#Credit [InsideNewCentury]
##########################################


from hello.dataWorld.audioTools import get_sig,audio_to_image
import soundfile as sf
import os
from tqdm import tqdm
import pandas as pd


from sklearn.model_selection import train_test_split
import shutil
    

AUDIO_COLS=["trkId","trkSig","trkSr","spId","spName","spSex","spAge","spRegion","spCity","recQuality"]
IMG_COLS=["imgId","imgSig","imgNFTT","imgWidth","imgHeight","spId","spName","spSex","spAge","spRegion","spCity","recQuality"]

def create_audio_df(path_dir,save_as=None,path_out=None):

    sp_dirs=os.listdir(path_dir)
    data=[]
    for sp_dir in sp_dirs:
        sp_row=sp_dir.split("_")
        print("Reading speaker:",sp_row)
        
       
        sp_tracks=os.listdir(path_dir+"/"+sp_dir)
        for f in tqdm(sp_tracks):
            trackId=f.split(".")[0].split("_")[-1]
            sig=get_sig(path=path_dir+"/"+sp_dir+"/"+f)
            data.append([trackId]+list(sig)+sp_row)

    print("Audio loaded, creating data frame...")
    df=pd.DataFrame(data,columns=AUDIO_COLS) 
    
    if save_as=="pickle":
        print(f"Saving in: {path_out} as: pickle")
        df.to_pickle(path_out)


    return df

def create_img_df(path_dir,save_as=None,path_out=None,width=64,height=64,n_ftt=4400):

    sp_dirs=os.listdir(path_dir)
    data=[]
    for sp_dir in sp_dirs:
        sp_row=sp_dir.split("_")
        print("Reading speaker:",sp_row[0])
        
       
        sp_tracks=os.listdir(path_dir+"/"+sp_dir)
        for f in tqdm(sp_tracks):
            trackId=f.split("_")[1].split(".")[0]
            sig=get_sig(path=path_dir+"/"+sp_dir+"/"+f)
            v,i=audio_to_image(sig=sig,scale="mel",width=width,height=height,n_fft=n_ftt) #np-array pil-img

            data.append([trackId,i,n_ftt,width,height]+sp_row)

    print("Audio loaded and converted into img, creating data frame...")
    df=pd.DataFrame(data,columns=IMG_COLS) 
    
    if save_as=="pickle":
        print(f"Saving in: {path_out} as: pickle")
        df.to_pickle(path_out)


    return df

def process_audio_fs(path_dir,path_out,trim_top_db=25):
    

    sp_dirs=os.listdir(path_dir)
    
    os.mkdir(path_out)
    for sp_dir in sp_dirs:
        path_spout=path_out+"/"+sp_dir
        os.mkdir(path_spout)
        print("Process speaker:",sp_dir)
        
       
        sp_tracks=os.listdir(path_dir+"/"+sp_dir)
        for f in tqdm(sp_tracks):
            path_sndout=path_spout+"/"+sp_dir+"_"+f.split("_")[1]
           
            
            sig=get_sig(path=path_dir+"/"+sp_dir+"/"+f,trim_top_db=trim_top_db)
            # Write out audio as 24bit PCM WAV
            sf.write(path_sndout, sig[0], sig[1], subtype='PCM_32') #subtype esportato da audacity, codifica
            

    print("Dataset created!")

def create_img_fs(path_dir,path_out,width=64,height=64,n_ftt=4400):
    
    
    
    sp_dirs=os.listdir(path_dir)
    
    os.mkdir(path_out)
    for sp_dir in sp_dirs:
        path_spout=path_out+"/"+sp_dir
        os.mkdir(path_spout)
        print("Process speaker:",sp_dir)
        
       
        sp_tracks=os.listdir(path_dir+"/"+sp_dir)
        for f in tqdm(sp_tracks):
            path_imgout=path_spout+"/"+f.split(".")[0]+".png"
           
            
            sig=get_sig(path=path_dir+"/"+sp_dir+"/"+f)
            v,i=audio_to_image(sig=sig,scale="mel",width=width,height=height,n_fft=n_ftt) #np-array pil-img
            i.save(path_imgout)

    print("Dataset created!")

def generate_deep_train_dataset(path_dir,path_out,shuffle=False,train_size=None,stratify=None,random_state=42):
    final=[]
    sps=os.listdir(path_dir)
    for spDir in sps:
        spFiles=os.listdir(path_dir+"/"+spDir)
        for spFile in spFiles:
            final.append((spDir+"/"+spFile,spFile))

    strat=None #TO DO estrarre la colonna categoria di riferimento a partire dai path e creare la array per lo stratify, aggiungere un plot dele distribuzioni delle laebl.

    trn,tst=train_test_split(final,train_size=train_size,shuffle=shuffle,random_state=random_state,stratify=strat)
    os.mkdir(path_out)
    os.mkdir(path_out+"/train")
    os.mkdir(path_out+"/val")
    #print(sps)
    for p in trn:
        pathF,pathT=path_dir+"/"+p[0],path_out+"/train/"+p[1]
        shutil.copy(pathF,pathT)
    for p in tst:
        pathF,pathT=path_dir+"/"+p[0],path_out+"/val/"+p[1]
        shutil.copy(pathF,pathT)
    print("Train len:",len(trn),"Validation len:",len(tst))
    return trn,tst


            


    