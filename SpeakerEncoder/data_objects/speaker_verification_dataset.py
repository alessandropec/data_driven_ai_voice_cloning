

from data_objects.speaker_data import Speaker
from data_objects.random_cycler import RandomCycler
from torch.utils.data import Dataset, DataLoader
from glob import glob
from pathlib import Path
from typing import List
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

class SpeakerBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int,window_size_s:int):
        self.speakers = speakers

        self.files = {s.file_name: s.random_partial(utterances_per_speaker) for s in speakers}


       
        self.window_size_s=window_size_s
        
        batch=[]
        self.sample_rates=[]
   
        for s in speakers:
            for u in self.files[s.file_name]:
                sig,sr=u.get_sig()
                self.sample_rates.append(sr)
                if self.window_size_s:            
                    batch.append(torch.Tensor(self.random_crop(sig,sr)))
                else:
                    self.batch.append(torch.Tensor(self.random_crop(sig)))
   


        if len(set(signal.shape[0] for signal in batch)) == 1:
            self.data = torch.stack(batch)
        else:
            self.data = pad_sequence(batch,batch_first=True)
    
    def random_crop(self,audio,sr):
        ws=self.window_size_s*sr
        if len(audio)<ws:
            diff = ws - len(audio)
            # Calcola l'inizio e la fine del padding
            start = diff // 2
            end = diff - start
            # Aggiungi il padding all'audio e restituiscilo
            return np.pad(audio, (start, end), 'constant')
        start = random.randint(0, len(audio) - ws)
        return audio[start:start+ws]



class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root: str,speakers_per_batch:int,
                utterances_per_speaker:int,multi_rec_sessions=False,model_sample_rate=16000):
        super().__init__()

        self.root = Path(datasets_root)
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        self.speakers_per_batch=speakers_per_batch
        self.utterances_per_speaker=utterances_per_speaker

        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")

        self.speakers = [Speaker(speaker_dir,utterances_per_speaker,multi_rec_sessions=multi_rec_sessions,model_sample_rate=model_sample_rate) for speaker_dir in speaker_dirs]
        self.n_samples=0
        if multi_rec_sessions:
            for s in self.speakers:
                for rs in s.rec_sessions:
                    self.n_samples+=len(list(rs.glob("*")))
        else:
            for s in self.speakers:
                path=self.root.joinpath(s.file_name)
                self.n_samples+=len(list(path.glob("*")))
                
        self.speaker_cycler = RandomCycler(self.speakers,speakers_per_batch)


    def __len__(self):
        return int(1e10)#Max number of speaker per batch
        
    def __getitem__(self, index):
      
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string



class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None,window_size_s=None):
        
        self.window_size_s=window_size_s
        self.utterances_per_speaker = dataset.utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=dataset.speakers_per_batch,
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=True, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker,self.window_size_s) 


