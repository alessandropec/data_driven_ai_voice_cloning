from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector,WavLMConfig
from .ge2e_net import GE2Enet
import torch

#Wav lm as speaker encoder
class WavLMSpeakerEncoder(GE2Enet):
    def __init__(self,device="cpu",pretrained_name="microsoft/wavlm-large",
                 all_hidden=False,activation_function=None,model_sample_rate=16000):
        super(WavLMSpeakerEncoder,self).__init__(device=device,activation_function=activation_function)
        self.pretrained_name=pretrained_name
        self.all_hidden=all_hidden
        self.model_sample_rate=model_sample_rate


        
        self.model=None
        # Network defition      
        #Preprocessor definition
        if not ("untrained" in pretrained_name):
            self.preprocessor=Wav2Vec2FeatureExtractor.from_pretrained(self.pretrained_name)
            self.model=WavLMForXVector.from_pretrained(self.pretrained_name)
         
            
        else:
            self.pretrained_name=self.pretrained_name.split("_")[0]
            config=WavLMConfig.from_pretrained(self.pretrained_name)
            self .preprocessor=Wav2Vec2FeatureExtractor(config)
            self.model=WavLMForXVector(config=config)
            print("Loaded untrained net with config:\n",config)
        
        self.model.to(self.device)
        self.to(self.device)


    
      

    def forward(self, inp):
        """
        Computes the embeddings of a batch of utterance wav.

        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
    
   
        preprocessed=self.preprocessor(inp,sampling_rate=self.model_sample_rate,return_tensors="pt",padding=True)["input_values"].squeeze(0).to(self.device)
        # Pass the preprocessed and fixed size input trough net

        

        

        
        embeds=self.model(preprocessed).embeddings

       
        # L2-normalize it
       
        embeds = embeds / (torch.norm(embeds, dim=-1, keepdim=True) + 1e-5)      

        return embeds
