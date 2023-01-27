# Data driven AI voice cloning

This repository is an implementation of the main part of my master thesis in Data science & Engineering.
It is divided in two part:

1. Speaker Encoder
  1. models: ECAPA-TDNN, wavlm-series
  2. data: VoxCeleb1, private dataset

2. Text-to-speech 
  1. model: FastSpeech2 (microsoft implementation)
  2. data: VCTK, LibriTTS

This two part are then integrated to achieve a Multi Speaker Text to Speech model that is capable of cloning unseen voices starting from about 5 seconds of audio, the ZeroShotFastSpeech2 model.

