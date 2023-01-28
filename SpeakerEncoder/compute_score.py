from model.ecapatdnn_speaker_encoder import EcapaTDNNSpeakerEncoder
from model.wavlm_speaker_encoder import WavLMSpeakerEncoder
import os
import audioTools
import torch
from tqdm import tqdm
import numpy as np
import argparse
from train import get_model

def get_pairs_list(pairs_list_path,data_root_path):
    data=[]
    with open(pairs_list_path) as fp:
        lines=fp.readlines()
        for l in lines:
            ls=l.strip().split(" ")
            ls[1]=os.path.join(data_root_path,ls[1])
            ls[2]=os.path.join(data_root_path,ls[2])
            data.append(ls)


    return data

# def get_model(model_name,device,load=False,load_in_gpu=False):
#     model=None
#     if model_name=="ecapa":
#         model=EcapaTDNNSpeakerEncoder(device)
#     if model_name=="wavlm-base":
#         model=WavLMSpeakerEncoder(device,pretrained_name="microsoft/wavlm-base")
#     if model_name=="wavlm-base-plus":
#         model=WavLMSpeakerEncoder(device,pretrained_name="microsoft/wavlm-base-plus")
#     if model_name=="wavlm-base-sv":
#         model=WavLMSpeakerEncoder(device,pretrained_name="microsoft/wavlm-base-sv")
#     if model_name=="wavlm-base-plus-sv":
#         model=WavLMSpeakerEncoder(device,pretrained_name="microsoft/wavlm-base-plus-sv")
#     if load:
#         map_location="cpu"
#         if load_in_gpu:
#             map_location="cuda"
#         if os.path.exists(load):         
#             print("Found existing model \"%s\", loading it." % load)
#             checkpoint = torch.load(load,map_location=torch.device(map_location))
#             model.load_state_dict(checkpoint["model_state"])
#         else:
#             print(f"Path not founded {load}")
        
#     return model

def compute_similarity_pairs(model,pairs_list_path,out_list_path,data_root_path,use_wb=False,):


    

    pairs=get_pairs_list(pairs_list_path=pairs_list_path,data_root_path=data_root_path)
    with torch.no_grad():
        with open(out_list_path,"w") as fp:
            for p in tqdm(pairs): 
                s1=audioTools.get_sig(p[1],sr=16000)[0]
                s2=audioTools.get_sig(p[2],sr=16000)[0]
                s1=torch.tensor(s1).unsqueeze(0).to(device)
                s2=torch.tensor(s2).unsqueeze(0).to(device)
                sim=model.compare_utterances(s1,s2,use_wb)
                outline=str(sim.item())+" "+" ".join(p)+"\n"
                fp.write(outline)

def compute_eer(model,resultPath):
    

    eer,fpr,tpr,tresholds=None,None,None,None

    with open(resultPath,"r") as fp:
        lines=fp.readlines()
        labels,preds=[],[]
        for l in tqdm(lines,desc="Computing Equal Error Rate"):
            array=l.split(" ")
            labels.append(int(array[1]))
            preds.append(float(array[0]))
        
        eer,fpr,tpr,thresholds=model.compute_eer(np.array(labels),np.array(preds))
        optimal_cutoff=model.find_optimal_cutoff(fpr,tpr,thresholds)

    return eer,fpr,tpr,thresholds,optimal_cutoff



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--compute_similarity_pairs",action="store_true")
    parser.add_argument("--compute_eer",action="store_true")
    parser.add_argument("--model_name")
    parser.add_argument("--enlarged_head",action="store_true")
    parser.add_argument("--load_path",default=False)
    parser.add_argument("--result_path")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--pairs_list_path",required=False)
    parser.add_argument("--data_root_path",required=False)
    parser.add_argument("--use_wb",default=False)
    parser.add_argument("--load_in_gpu",action="store_true")
    

    args=parser.parse_args()
    args_keys=list(vars(args).keys())

    device=torch.device(args.device)
    model=get_model(args.model_name,device=device,activation_function=None,train_only_head=False,enlarge_head=args.enlarged_head)
    
    if args.load_path!=False:
        map_location="cpu"
        if args.load_in_gpu:
            map_location="cuda"
        if os.path.exists(args.load_path):         
            print("Found existing model \"%s\", loading it." % args.load_path)
            checkpoint = torch.load(args.load_path,map_location=torch.device(map_location))
            model.load_state_dict(checkpoint["model_state"])
        else:
            print(f"Path not founded {args.load_path}")

    if args.compute_similarity_pairs:
        if (not "pairs_list_path" in args_keys) or (not "data_root_path" in args_keys):
            print("Error you need to specify the path of the pairs list")
            exit()
        compute_similarity_pairs(model,args.pairs_list_path,
                                 args.result_path,args.data_root_path,args.use_wb)
    if args.compute_eer:
        outs=compute_eer(model,args.result_path)
        print(outs)