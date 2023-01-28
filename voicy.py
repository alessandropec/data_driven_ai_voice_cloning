import sys
sys.path.append("SpeakerEncoder")
sys.path.append("ZeroShotFastSpeech2")

from SpeakerEncoder import get_speaker_emb,get_speaker_model
import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from ZeroShotFastSpeech2.utils.model import get_model as get_synth_model, get_vocoder
from ZeroShotFastSpeech2.utils.tools import to_device, synth_samples
from ZeroShotFastSpeech2.dataset import TextDataset
from ZeroShotFastSpeech2.text import text_to_sequence
from ZeroShotFastSpeech2.synthesize import preprocess_english,preprocess_mandarin,synthesize

import yaml
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args():
    def __init__(self,conf) -> None:
        self.preprocess_config=conf["prepro"]
        self.train_config=conf["train"]
        self.model_config=conf["model"]
        self.mode=conf["mode"]
        self.source=conf["source"]
        self.speaker_input=conf["speaker_input"]


if __name__=="__main__":
    speaker_model=get_speaker_model("ecapa","cpu",None)
    speaker_emb=get_speaker_emb("ZeroShotFastSpeech2/datasets/LibriTTS/8887/281471/8887_281471_000000_000000.wav",speaker_model)
    args=Args(
        {
        "prepro":"./ZeroShotFastSpeech2/config/LibriTTS/preprocess.yaml",
        "train":"./ZeroShotFastSpeech2/config/LibriTTS/train.yaml",
        "model":"./ZeroShotFastSpeech2/config/LibriTTS/model.yaml",
        "mode":"single",
        "source":"Hi, my name is Peterson",
        "speaker_input":speaker_emb
        
        }
    )
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    preprocess_config["path"]={key:os.path.join("ZeroShotFastSpeech2",path) for key,path in preprocess_config["path"].items()}
    #train_config["path"]=[os.path.join("ZeroShotFastSpeech2",path) for path in model_config["path"]]
    model_config["path"]={key:os.path.join("ZeroShotFastSpeech2",path) for key,path in train_config["path"].items()}


    configs = (preprocess_config, model_config, train_config)



    # Get model
    model = get_synth_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_input])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        #
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)