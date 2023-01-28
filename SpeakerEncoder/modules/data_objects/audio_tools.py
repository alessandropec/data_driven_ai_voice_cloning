import librosa,librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import pyaudio
import wave
from IPython.display import Audio as printAudio
from PIL import Image
import speech_recognition as sprec
import io
from pydub import AudioSegment

import numpy as np
from scipy.io.wavfile import read
import torch
import glob
import os


def normalize(audio):
    
    max_abs=torch.max(torch.max(audio),torch.abs(torch.min(audio)))
    audio= audio/max_abs
    return audio

def trim_silence(audio,start_threshold=0.00001,end_threshold=0.00001):
    energy=audio.pow(2)
    # Find the start and end indices where the energy is above a threshold

    start_idx=0
    if start_threshold:
        start_idx = (energy > start_threshold).nonzero().min()

    end_idx=len(audio)
    if end_threshold:
        end_idx=(energy > end_threshold).nonzero().max()

    audio=audio[start_idx:end_idx]

    return audio

def add_silence(audio,sr,duration_in_seconds=0.02,end=True,start=False):
    num_samples = int(duration_in_seconds * sr)
    silence = torch.zeros((num_samples))

    if end:
        # Concatenate the audio and silence
        audio = torch.cat((audio, silence), dim=0)
    if start:
        audio=torch.cat((silence,audio),dim=0)

    return audio

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_vctk(root_path,wav_folder="wav",text_folder="txt"):
    '''For vctk dataset'''
    wav_path=os.path.join(root_path,wav_folder)
    text_path=os.path.join(root_path,text_folder)
    txts_p=glob.glob(text_path+"/*/*")
    wavs_p=glob.glob(wav_path+"/*/*")

    wavs_p=[w for w in wavs_p if ".wav" in w]
    return list(zip(wavs_p,txts_p))

def load_libri_tts(root_path,normalized_text=True):
    '''For LibriTTS dataset'''
    all_files=glob.glob(root_path+"/*/*/*")
    
    if normalized_text:
        normalized_text=".normalized"
    else:
        normalized_text=".original"
    wavs_p=[p for p in all_files if ".wav" in p]
    txts_p=[p for p in all_files if normalized_text in p]
    

    return list(zip(wavs_p,txts_p))


#-----------------------------Signal/Audio processing

def get_sig(path,trim_top_db=None,sr=22050):
    '''
    Read an audio from different format, see librosa.load()

    path: string - path to file
    trim_top_db: None | int - If int trim audio, see librosa.effects.trim()

    RETURN: tuple(signal: np array, int: sample rate)
    '''
    sign,sr=librosa.load(path,sr=sr)

    if trim_top_db!=None:
        sign, _ = librosa.effects.trim(y=sign,top_db=trim_top_db) #return signal,index of trimming

    return  (sign,sr)#sig,sr
    #print("Signal shape",signal_2.shape,"Sample rate",sr)

def get_fft(sig=None,path=None):
        # Creating a Discrete-Fourier Transform with our FFT algorithm
    if(path!=None):
        sig=get_sig(path)

    wav,sr=sig
    
    fast_fourier_transf_1 = np.fft.fft(wav)
    
    # Magnitudes indicate the contribution of each frequency
    magnitude_1 = np.abs(fast_fourier_transf_1)

    # mapping the magnitude to the relative frequency bins
    frequency_1 = np.linspace(0, sr, len(magnitude_1))
    
    # We only need the first half of the magnitude and frequency
    left_mag_1 = magnitude_1[:int(len(magnitude_1)/2)]
    left_freq_1 = frequency_1[:int(len(frequency_1)/2)]

    return (left_freq_1,left_mag_1)

def get_spectr(sig=None,path=None,n_fft=2048,hop_length=512,win_len=1024,scale="amp"):
    # n_fft this is the number of samples in a window per fft 
    # hop_length The amount of samples we are shifting after each fft
    if(path!=None):
        sig=get_sig(path)

    wav,sr=sig

    if scale=="mel":
        mel_signal_1 = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length, n_fft=n_fft,win_length=win_len)
        mel_spect_1 = np.abs(mel_signal_1)
        power_to_db_1 = librosa.power_to_db(mel_spect_1, ref=np.max)
        return power_to_db_1
    else:

        # Short-time Fourier Transformation on our audio data
        audio_stft_1 = librosa.core.stft(wav, hop_length=hop_length, n_fft=n_fft)
        # gathering the absolute values for all values in our audio_stft
        spectrogram_1 = np.abs(audio_stft_1)

        if scale=="amp":
            return spectrogram_1
        elif scale=="db":
            return librosa.amplitude_to_db(spectrogram_1)
    
   
    
    #(Magnitude shape)/(n_ftt-hop_length) campioni iniziali / ampiezza window - grandezza shift

def listen(callable_fn=lambda v: print("Sound detected."),sample_rate=16000,verbose=True):
    s_rec=sprec.Recognizer()
    #if verbose: print("I'm listening...")
    with sprec.Microphone(sample_rate=sample_rate) as source:
        while True:
            if verbose:
                print("I'm listening...")
            audio=s_rec.listen(source) #pyaudio obj
            if verbose: print("Sound detected and recorded.")
            data_wav=io.BytesIO(audio.get_wav_data()) #byte stream
            clip_wav=np.array(AudioSegment.from_wav(data_wav).get_array_of_samples(),dtype=float) #numpy obj
            sig=(clip_wav,sample_rate)
            callable_fn(sig)

def record(seconds=5,path="trim_experiment.wav",ch=1):
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = ch
    RATE = 44100
    RECORD_SECONDS = seconds
    WAVE_OUTPUT_FILENAME = path

    p = pyaudio.PyAudio()

    SPEAKERS = p.get_default_output_device_info()["hostApi"] 
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_host_api_specific_stream_info=SPEAKERS,
                    ) 

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#---------------------------------------Image processing

def audio_to_image(sig,n_fft=4410,scale="mel",num_channels=3,win_sizes=[25,50,100],hop_sizes=[10,25,50],height=128,width=250):
    #TO DO: aggiungere un check sul resize non ha senso aumentare i pixel
    _,sr=sig

    specs = []
    for i in range(num_channels):
        window_length = int(round(win_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))

        #clip = torch.Tensor(wav)
        
        #spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=4410, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
        #eps = 1e-6 TO DO: capire meglio
        #spec = spec.numpy()
        #spec = np.log(spec+ eps)

        spec= get_spectr(sig,hop_length=hop_length,scale=scale,win_len=window_length,n_fft=n_fft)
        spec = np.asarray(torchvision.transforms.Resize((height, width))(Image.fromarray(spec)))
        specs.append(spec)

    arrayRGB=np.stack(specs,axis=-1)
    imgRGB=Image.fromarray(arrayRGB,"RGB")
    return arrayRGB,imgRGB



#--------------------------------Signal Viz
def plot_wav(sig,figsize=(20,5)):

    wav,sr=sig
    plt.figure(figsize=figsize)
    librosa.display.waveshow(wav, sr=sr)
    plt.title('Waveplot', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Amplitude', fontdict=dict(size=15))
    

def plot_fft(fft):
    freq,magnitude=fft
    plt.figure(figsize=(20, 5))
    plt.plot(freq, magnitude)
    plt.title("Discrete-Fourier Transform", fontdict=dict(size=15))
    plt.xlabel("Frequency", fontdict=dict(size=12))
    plt.ylabel("Magnitude", fontdict=dict(size=12))

def plot_spectr(spectr,sr,hop_length=512,scale_name="amp",figsize=(20,5)):

    # Plotting the short-time Fourier Transformation
    plt.figure(figsize=figsize)
    # Using librosa.display.specshow() to create our spectrogram
    librosa.display.specshow(spectr, sr=sr, x_axis='time',        y_axis='hz', cmap='magma', hop_length=hop_length)
    plt.colorbar(label=scale_name)
    plt.title('Spectrogram: '+ scale_name, fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    




    




 