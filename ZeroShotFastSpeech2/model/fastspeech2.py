import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

def speaker_adding(output,speakers,max_src_len,speaker_projector,speaker_adding_strategy):
    if speaker_projector:
        speakers=speaker_projector(speakers)

   
    if speaker_adding_strategy=="concat":

        speakers_embedding=speakers.unsqueeze(1).repeat(1,output.size(1),1)
   
        output = torch.cat(
                (output, speakers_embedding), dim=2)
    elif speaker_adding_strategy=="sum":
          
            prepare_speakers=speakers.unsqueeze(1).expand(
        -1, max_src_len, -1)
            output = output + prepare_speakers
    

    return output



class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        
        self.compute_speaker_emb = None
        
        self.use_speaker_emb=preprocess_config["speaker_emb"]

        self.speaker_projector=None
        if self.use_speaker_emb:
            self.speaker_adding_strategy=model_config["speaker_adding_strategy"]
            self.speaker_adding_location=model_config["speaker_adding_location"]

        enlarged_dim=None
        if model_config["multi_speaker"] and not self.use_speaker_emb:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.compute_speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        elif model_config["multi_speaker"] and self.use_speaker_emb: #Change input encoder dim, another way is use a linear layer
            if model_config["speaker_adding_strategy"]=="concat":
                enlarged_dim=model_config["transformer"]["encoder_hidden"]
            
                if model_config["speaker_projector_dim"]>0:
                    self.speaker_projector=torch.nn.Linear(model_config["speaker_emb_dim"],
                                                        model_config["speaker_projector_dim"])
                    enlarged_dim=enlarged_dim+model_config["speaker_projector_dim"]
                else:    
                    enlarged_dim=enlarged_dim+model_config["speaker_emb_dim"]

            #Set to projection to match the decoder dim
            elif model_config["speaker_adding_strategy"]=="sum":
                print("Speaker adding strategy, sum, using speaker projector!")
                enlarged_dim= model_config["transformer"]["encoder_hidden"]
                model_config["speaker_projector_dim"]=enlarged_dim
                self.speaker_projector=torch.nn.Linear(model_config["speaker_emb_dim"],
                                                    model_config["speaker_projector_dim"])

            #if model_config["speaker_adding_location"]!="pre_variance_adaptor":
            model_config["transformer"]["decoder_hidden"]=enlarged_dim
            print("Decoder dimension:", model_config["transformer"]["decoder_hidden"])

        self.model_config = dict(model_config)

        enlarge_variance_dim=0
        if model_config["speaker_adding_location"]=="pre_variance_adaptor" and model_config["speaker_adding_strategy"]=="concat":
            if model_config["speaker_projector_dim"]>0:
                enlarge_variance_dim=model_config["speaker_projector_dim"]
            else:
                enlarge_variance_dim=model_config["speaker_emb_dim"]
        print("Enlarge variance adaptor dim of:", enlarge_variance_dim)
        
        self.encoder = Encoder(self.model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, self.model_config,enlarge_variance_dim)
        self.decoder = Decoder(self.model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()


    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        

        if self.compute_speaker_emb is not None:
            output = output + self.compute_speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        if self.use_speaker_emb and self.speaker_adding_location=="pre_variance_adaptor":
            output=speaker_adding(output,speakers,max_src_len,self.speaker_projector,self.speaker_adding_strategy)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        ##################Speaker adding############
        if self.use_speaker_emb and self.speaker_adding_location=="post_variance_adaptor":
            output=speaker_adding(output,speakers,output.size(1),self.speaker_projector,self.speaker_adding_strategy)
           


        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )