import argparse
import torch
from data_objects.audio_tools import normalize,trim_silence,load_wav_to_torch,load_libri_tts,load_vctk
import librosa
from tqdm import tqdm
import os
import glob
from train import get_model

def get_speaker_model(model_name="ecapa",device="cpu",ckpt_path=None):
        
        
        enlarge_head=False
        if "wavlm-large" in model_name and "big_head" in model_name:
            enlarge_head=True
        model=None
        with torch.no_grad():
            model=get_model(model_name=model_name,device=device,enlarge_head=enlarge_head,activation_function=None,train_only_head=False)

            if ckpt_path:              
                print("Found existing model \"%s\", loading it and resuming training." % ckpt_path)
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint["model_state"]),
            model.train(False)
            return model
        
def get_spaekr_emb(filename,speaker_model):
    audio,sampling_rate=load_wav_to_torch(filename)
    audio_norm=normalize(audio)
    audio_trim = trim_silence(audio_norm,start_threshold=start_threshold,end_threshold=end_threshold)

    if sampling_rate!=48000 and sampling_rate!=16000:
    
    
    
        sp_audio,sp_sr=torch.tensor(librosa.resample(audio_trim.numpy(),sampling_rate,16000)),16000
    else:
        sp_audio,sp_sr=audio_trim,48000
    emb=speaker_model(sp_audio.unsqueeze(0))

    return emb

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default=None,
                        help='ecapa or wavlm series')
    parser.add_argument('--checkpoint', default=None,required=False,
                        help='load a saved speaker verification model')
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--dataset_dir",type=str)
    parser.add_argument("--model_device",type=str)

    
 

    

    args = parser.parse_args()
    print(parser)
    print(args)

    print(args.model_name)


    speaker_model=get_speaker_model(model_name=args.model_name,device=args.model_device,
                                            ckpt_path=args.checkpoint)
    sampling_rate=None
    start_threshold,end_threshold=None,None
    speaker_folder=[]
    if args.dataset_name=="VCTK":
        print("#################-Loading VCTK paths-##########################")
        file=load_vctk(args.dataset_dir)
        sampling_rate=48000
        start_threshold,end_threshold=0.01,0.05
        wavs=[f[0] for f in file]

        
        for f in file:
            tmp=os.path.split(f[0])[0] #remove file
            speaker_folder.append(tmp)

    elif args.dataset_name=="LibriTTS":
        print("#################-Loading libri tts paths-##########################")
        file=load_libri_tts(args.dataset_dir)
        start_threshold,end_threshold=0.0001,0.0001
        sampling_rate=24000


        for f in file:
            tmp=os.path.split(f[0])[0] #remove file
            tmp=os.path.split(tmp)[0] #remove libri folder
            speaker_folder.append(tmp)
        

   
    speaker_folder=set(speaker_folder)
    print("Number of speaker:",len(speaker_folder))


    files=None
    with torch.no_grad():
        print("#########-START CREATION EMBEDDINGS-########")

        for speaker in tqdm(speaker_folder):

            if args.dataset_name=="VCTK":
                files=[f for f in glob.glob(speaker+"/*") if ".wav" in f]
            
            elif args.dataset_name=="LibriTTS":
                files=[f for f in glob.glob(speaker+"/*/*") if ".wav" in f]
      
            
            embs=[]
            for filename in tqdm(files):
                try:
                    emb=get_spaekr_emb(filename,speaker_model)

                    emb_name=os.path.splitext(filename)[0]
                    emb_name+="."+args.model_name+"_embedding"
                    torch.save(emb,emb_name)
                    embs.append(emb)

                except Exception as e:
                    print(e,filename)
            embs=torch.stack(embs)
            embs=torch.mean(embs,dim=0)
            avg_emb_path=os.path.join(speaker,f"{os.path.split(speaker)[1]}.{args.model_name}_averaged_embedding")
            torch.save(embs,avg_emb_path)

