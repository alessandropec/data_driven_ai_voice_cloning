# Speaker Encoder

This folder contain all stuff to train different speaker verification model with the [Generalized end-to-end loss](https://arxiv.org/pdf/1710.10467.pdf)
The code provide a wrapper that use the GE2E loss for all model proposed that are pretrained from Hugging Face and are:

1. [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
2. [WAVLM-series](https://huggingface.co/spaces/microsoft/wavlm-speaker-verification)

## Training

Training log of speaker encoder is done with [CometML](https://www.comet.com/site/) you need to register on the site and get the api key.
All parameters are inside speaker train_config.json, set it and run:

```
python train_encoder.py --config=train_config.json
```
### Non obvious parameters in train_config.json



## Embedding generation

To generate embedding for VCTK or LIbriTTS dataset (or in general for any dataset that follow VCTK or LibriTTS file system structure), you can run

```
python create_speaker_emb_dataset.py --model_name ecapa --dataset_name VCTK --dataset_dir ROOT_PATH_OF_DATASET --model_device cuda --checkpoint model.pt
```

1. model_name: **ecapa**, **wavlm-base-plus-sv**, **wavlm-large**, wavlm-base, wavlm-base-sv
  Specify one of the above name to start from pretraining model or specify one of the above name and the checkpoint to use your trained model
2. dataset_name: VCTK, LibriTTS
  Specify the file system structure of your dataset
3. dataset_dir: the root path of the data
4. model_device: the creation can be run both on cpu or cuda
5. checkpoint: path to your checkpoint, pay attention to align the model name (mdoel architecture) with the right checkpoint (model weight)

The script create for each audio in the dataset the corresponding embedding in the folder where is the audio, and for each speaker the averaged embedding in the folder of the speaker, the names of the embedding file are the following:
``` audioName.modelName_embedding``` for single audio embedding and ```speakerName.modelName_averaged_embedding ``` for speaker averaged embedding
you can find some example [here](https://github.com/alessandropec/data_driven_ai_voice_cloning/tree/master/ZeroShotFastSpeech2/datasets/LibriTTS/8887)
