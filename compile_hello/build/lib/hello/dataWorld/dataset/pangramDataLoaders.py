from torch.utils.data import Dataset,DataLoader
import os
from hello.dataWorld.audioTools import get_sig
import PIL

import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence

class SpeakerSignalDataset(Dataset):

    def __init__(self, files_dir="./speaker_classifier_data",signalType="png",prepro_func=None):

        super().__init__()
        self.files_dir = files_dir
        self.files_name = [os.path.join(self.files_dir,fn) for fn in os.listdir(files_dir)]    
        self.signalType=signalType

        
      
        #self.labels=[int(os.path.split(p)[1].split("_")[0]) for p in self.files_name]
        self.names=[str(os.path.split(p)[1].split("_")[1]) for p in self.files_name]

        lab=list(set(self.names))
        self.id2label={}
        self.label2id={}

        for i,l in enumerate(lab):

          self.id2label[i]=l
          self.label2id[l]=i




        self.labels=[]
        for l in self.names:
          self.labels.append(tensor(self.label2id[l]))


                

        self.prepro_func=prepro_func

    #Legge un audio da cartella Ritorna lo spettrogramma di mel
    def __getitem__(self, idx):
        signal_path = self.files_name[idx]

        if self.signalType in ["png","jpg"]: #see pil image open
            if self.prepro_func:
                return self.prepro_func((PIL.Image.open(signal_path),self.labels[idx]))
            else:
                return PIL.Image.open(signal_path),self.labels[idx]
        elif self.signalType=="wav":
            if self.prepro_func:
                return self.prepro_func((get_sig(signal_path),self.labels[idx]))
            else: 
                return tensor(get_sig(signal_path)[0]),self.labels[idx]

    def __len__(self):
        return len(self.files_name)

def get_SpeakerSignalDataLoader(files_dir,signalType,prepro_func, batch_size, num_workers,collate_fn=None):
	dataset = SpeakerSignalDataset(files_dir,signalType,prepro_func=prepro_func)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,collate_fn=collate_fn)
	return dataloader,dataset

def pad_collate(batch):
    #Remove sr in position 1
  
    (xx, yy) = zip(*batch)
    #print(xx[3],xx[3].shape)
    x_lens = [len(x) for x in xx]
    y_lens = [1 for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    #yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, torch.flatten(torch.tensor(yy)), x_lens, y_lens

        
    
        
       


