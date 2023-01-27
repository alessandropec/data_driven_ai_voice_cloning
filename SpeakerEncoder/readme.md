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
#### Non obvious parameters in train_config.json

1. **model_name**: **ecapa**, **wavlm-base-plus-sv**, **wavlm-large**, wavlm-base, wavlm-base-sv.
2. **run_id**: name of the run on comet.
3. **multi_rec_session**: set to True if inside the speaker folder there are different folder, like VoxCeleb or LibriTTS where each folder is a video or a book.
4. **model_sample_rate**: to align the input audio sample rate to the sample rate of the audios from which the model is pretrained (16Khz).
5. **window_size_s**: during training tha audio is random cropped and centered, you can specify the length of the crop in seconds.
6. **vis_train_art_every**: plot umap of the embedding and store the audios in input, you can specify the interval.
7. **vis_train_metrics_every**: log the loss and the metrics.
8. **val_every**: done validation every.
9. **vis_val_art_every**: plot umap of the embedding and store the audios in input during validation, you can specify the interval.
10. **annote_point**: add for each point in the scatter the name of the audio.
11. **val_max_epochs**: specify how much epoch iteration on validation .
12. **use_data_augmentation**: use torch-audiomentation to do data augmentation.
13. **augment_noise_paths**: ["PATH_TO_EACH_FOLDER_CONTAINS_NOISE1","PATH_TO_EACH_FOLDER_CONTAINS_NOISE2","ETC..."]
14. **augment_musics_paths**: as above.
15. **augment_reverbs_paths**: as above.
16. **data_parallel**: specify the device ids if you run on multi gpu, set to [0] to run on single gpu.



## Speaker embedding creation

To generate embedding for VCTK or LIbriTTS dataset (or in general for any dataset that follow VCTK or LibriTTS file system structure), you can run

```
python create_speaker_emb_dataset.py --model_name ecapa --dataset_name VCTK --dataset_dir ROOT_PATH_OF_DATASET --model_device cuda --checkpoint model.pt
```

1. **model_name**: **ecapa**, **wavlm-base-plus-sv**, **wavlm-large**, wavlm-base, wavlm-base-sv
>Specify one of the above name to start from pretraining model or specify one of the above name and the checkpoint to use your trained model
2. **dataset_name**: VCTK, LibriTTS
>Specify the file system structure of your dataset
3. **dataset_dir**: the root path of the data
4. **model_device**: the creation can be run both on cpu or cuda
5. **checkpoint**: path to your checkpoint, pay attention to align the model name (mdoel architecture) with the right checkpoint (model weight)

The script create for each audio in the dataset the corresponding embedding, in the folder where the audio is already located, moreover for each speaker the averaged embedding is computed and stored in the folder of the speaker. The names of the embedding file are the following:
``` audioName.modelName_embedding``` for single audio embedding and ```speakerName.modelName_averaged_embedding ``` for speaker averaged embedding
you can find some example [here](https://github.com/alessandropec/data_driven_ai_voice_cloning/tree/master/ZeroShotFastSpeech2/datasets/LibriTTS/8887).

## Umap of some speaker embeddings from VoxCeleb 1 with ECAPA-TDNN
![umap of speakers from VoxCeleb1](https://github.com/alessandropec/data_driven_ai_voice_cloning/blob/master/SpeakerEncoder/vox1_umap_example.png)
