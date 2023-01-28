#Ecapa tdnn as speaker encoder
from .ge2e_net import GE2Enet
from speechbrain.pretrained import EncoderClassifier
import torch


class EcapaTDNNSpeakerEncoder(GE2Enet):

    def __init__(self,device="cpu",pretrained_name="speechbrain/spkrec-ecapa-voxceleb",activation_function=None):
        super(EcapaTDNNSpeakerEncoder,self).__init__(device=device,activation_function=activation_function)
        self.pretrained_name=pretrained_name
        self.model=None
        if pretrained_name:
            if device=="cuda":
                
                self.model=EncoderClassifier.from_hparams(source=self.pretrained_name,run_opts={"device":"cuda"})
            else:
                self.model=EncoderClassifier.from_hparams(source=self.pretrained_name)
            print(self.pretrained_name,"loaded pretrained")
        else:           
            print("Not implemented")
            raise Exception

        

        self.to(self.device)

    def forward(self,inp):
    
          
        embeddings=self.model.encode_batch(inp)

        
        #Remove useless dimension and normalize
        return (embeddings / (torch.norm(embeddings, dim=-1, keepdim=True) + 1e-5)).squeeze(1)

   

        
