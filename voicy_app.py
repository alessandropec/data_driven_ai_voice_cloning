import sys
sys.path.append("SpeakerEncoder")
sys.path.append("ZeroShotFastSpeech2")
from SpeakerEncoder import get_speaker_emb,get_speaker_model
from ZeroShotFastSpeech2.utils.tools import to_device, synth_samples
from ZeroShotFastSpeech2.text import text_to_sequence
from ZeroShotFastSpeech2.synthesize import preprocess_english,synthesize
from ZeroShotFastSpeech2.utils.model import get_model as get_synth_model, get_vocoder


from PIL import ImageTk,Image

import torch
import numpy as np
import umap
import time

import matplotlib.pyplot as plt
import os

import yaml


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import playsound


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH=False
class Args():
    def __init__(self,conf) -> None:
        self.preprocess_config=conf["prepro"]
        self.train_config=conf["train"]
        self.model_config=conf["model"]
        self.mode=conf["mode"]
        self.text=conf["text"]
        self.speaker_input=conf["speaker_input"]
        self.restore_step=conf["restore_step"]
        self.pitch_control=conf["pitch_control"]
        self.energy_control=conf["energy_control"]
        self.duration_control=conf["duration_control"]




class SoundApp(tk.Frame):
    def __init__(self,speaker_model, master=None):
        super().__init__(master)
        self.ckpt_path="ZeroShotFastSpeech2/output_prcs_large"
        self.ckpt_step=8500

        self.speaker_model=speaker_model
     
        self.master = master# 1 riga due colonne 

        #self.master.grid_rowconfigure(1,weight=1) #embedding
        self.master.grid_rowconfigure(0,weight=1) #content_list
        self.master.grid_columnconfigure(0,weight=0) #content_list
        self.master.grid_columnconfigure(1,weight=4) #content_list
        #self.master.grid_columnconfigure(1,weight=1)
        

        self.items=[]

        self.browse_folder_path=None

        self.embeddings_fig=None

        self.spectro_widget=None

        self.synth_file_name=None

        self.populate_content_list()

        self.populate_content_synth()

     

    def populate_content_list(self,blank_figure=True):
        self.content_list = tk.Frame(self.master) #1 colonna 4 righe
        self.content_list.grid_rowconfigure(0,weight=0) #browse button
        self.content_list.grid_rowconfigure(1,weight=2) #browse content
        self.content_list.grid_rowconfigure(2,weight=2) #embedding
        self.content_list.grid_rowconfigure(3,weight=0) #display
        self.content_list.grid_columnconfigure(0,weight=1)
        self.content_list.grid(row=0, column=0,sticky="news")

        self.menu_frame=tk.Frame(self.content_list)
        self.menu_frame.grid(row=0,column=0)

        self.browse_button = tk.Button(self.menu_frame, text="Browse wav file", command=self.browse_folder)
        self.browse_button.grid(row=1, column=0, sticky="nwe")

        self.synth_model_label=tk.Label(self.menu_frame,text="Enter the path of the output of synth train")
        self.synth_model_label.grid(row=0,column=0,sticky="ne")
        self.synth_model_entry = tk.Entry(self.menu_frame)
        self.synth_model_entry.insert(0,self.ckpt_path)
        self.synth_model_entry.grid(row=0, column=1, sticky="nwe")

        self.synth_ckpt_label=tk.Label(self.menu_frame,text="Enter the ckpt step of the synth model to load")
        self.synth_ckpt_label.grid(row=0,column=2,sticky="ne")
        self.synth_ckpt_entry = tk.Entry(self.menu_frame)
        self.synth_ckpt_entry.insert(0,self.ckpt_step)
        self.synth_ckpt_entry.grid(row=0, column=3, sticky="nwe")


        self.browse_frame=tk.Frame(self.content_list)
        self.browse_frame.grid(row=1, column=0,sticky="new")

        self.browse_frame.grid_columnconfigure(0,weight=1)
        self.browse_frame.grid_rowconfigure(0,weight=1)
        
        self.list_canvas = tk.Canvas(self.browse_frame)
        self.list_canvas.grid_columnconfigure(0,weight=1)
        self.list_canvas.grid(row=0, column=0,sticky="new")

        self.list_frame = tk.Frame(self.list_canvas)
        self.list_frame.pack()

        self.list_scrollbar = tk.Scrollbar(self.browse_frame, orient="vertical", command=self.list_canvas.yview)
        self.list_scrollbar.grid(row=0, column=0, sticky="nse")

        self.list_canvas.configure(yscrollcommand=self.list_scrollbar.set)
        self.list_canvas.create_window((0,0), window=self.list_frame, anchor="nw")

        self.list_frame.bind("<Configure>", self.on_frame_configure)


        
     
        
        
        if blank_figure:
            self.embeddings_fig, ax = plt.subplots()
            self.embeddings_fig.figsize=(1, 3)
            self.embeddings_fig.suptitle("Selected speaker embeddings umap")

        self.figure_widget = FigureCanvasTkAgg(self.embeddings_fig, self.content_list)
        self.figure_widget.get_tk_widget().grid(row=2, column=0,sticky="ew")
        #self.figure_widget.get_tk_widget().config(width=self.content_list.winfo_width())
        self.figure_widget.draw()

        self.display_button = tk.Button(self.content_list, text="Display embeddings", command=self.display_embeddings)
        self.display_button.grid(row=3, column=0, sticky="we")

        



    def on_frame_configure(self, event):
        self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all"))

   
    def populate_content_synth(self):
        self.wrapper=tk.Frame(self.master)
        self.wrapper.grid(row=0,column=1,sticky="news",ipadx=50)
        
        self.wrapper.grid_rowconfigure(0,weight=0)#File name
        self.wrapper.grid_rowconfigure(1,weight=0)# control
        self.wrapper.grid_rowconfigure(2,weight=0)#Label input
        self.wrapper.grid_rowconfigure(3,weight=0) # Text Input
        self.wrapper.grid_rowconfigure(4,weight=0)#Synth button
        self.wrapper.grid_rowconfigure(5,weight=5)# Graph spectrogram
        self.wrapper.grid_rowconfigure(6,weight=1)# Synthetized audio
        self.wrapper.grid_columnconfigure(0,weight=1)# only one column
        


 
        #File name
        self.wrapper_file_name=tk.Frame(self.wrapper)
        self.wrapper_file_name.grid(row=0,column=0,sticky="nwse")
        self.wrapper_file_name.grid_columnconfigure(0,weight=0) #Label
        self.wrapper_file_name.grid_columnconfigure(1,weight=2) #Input

        self.filename_label = tk.Label(self.wrapper_file_name, text="Synthetized file name:")
        self.filename_label.grid(row=0, column=0,sticky="nw")

        self.filename_entry = tk.Entry(self.wrapper_file_name)
        self.filename_entry.insert(0,"synth_voice_test")
        self.filename_entry.grid(row=0, column=1,sticky="we")

        #Control feature
        self.wrapper_control=tk.Frame(self.wrapper)
        self.wrapper_control.grid(row=1,column=0,sticky="nwse")
        self.wrapper_control.grid_columnconfigure(0,weight=1) #ptich
        self.wrapper_control.grid_columnconfigure(1,weight=1) #duration
        self.wrapper_control.grid_columnconfigure(2,weight=1) #energy

        #Pitch
        self.pitch_frame=tk.Frame(self.wrapper_control)
        self.pitch_frame.grid_rowconfigure(0,weight=0)
        self.pitch_frame.grid_rowconfigure(1,weight=1)
        self.pitch_frame.grid_columnconfigure(0,weight=0)
        self.pitch_frame.grid(row=0,column=0,sticky="news")
     

        self.pitch_label = tk.Label(self.pitch_frame, text="Pitch control:")
        self.pitch_label.grid(row=0, column=0,sticky="news")
        self.pitch_entry=tk.Entry(self.pitch_frame)
        self.pitch_entry.insert(0,1.0)
        self.pitch_entry.grid(row=1, column=0,sticky="news")

        #Duration
        self.duration_frame=tk.Frame(self.wrapper_control)
        self.duration_frame.grid_rowconfigure(0,weight=0)
        self.duration_frame.grid_rowconfigure(1,weight=1)
        self.duration_frame.grid_columnconfigure(0,weight=0)
        self.duration_frame.grid(row=0,column=1,sticky="news")
        
        self.duration_label = tk.Label(self.duration_frame, text="Duration control:")
        self.duration_label.grid(row=0, column=0,sticky="news")
        self.duration_entry=tk.Entry(self.duration_frame)
        self.duration_entry.insert(0,1.0)
        self.duration_entry.grid(row=1, column=0,sticky="news")

        #Energy
        self.energy_frame=tk.Frame(self.wrapper_control)
        self.energy_frame.grid_rowconfigure(0,weight=0)
        self.energy_frame.grid_rowconfigure(1,weight=1)
        self.energy_frame.grid_columnconfigure(0,weight=0)
        self.energy_frame.grid(row=0,column=2,sticky="news")

        self.energy_label = tk.Label(self.energy_frame, text="Energy control:")
        self.energy_label.grid(row=0, column=0,sticky="news")
        self.energy_entry=tk.Entry(self.energy_frame)
        self.energy_entry.insert(0,1.0)
        self.energy_entry.grid(row=1, column=0,sticky="news")




        #Input label
        self.input_label = tk.Label(self.wrapper, 
                                     text="Select one or more wav, enter text to synthetize and click on synthetize!")
        self.input_label.grid(row=2, column=0, sticky="we")

        #Input text
        self.input_text = tk.Text(self.wrapper,height=5)
        self.input_text.insert(1.0,"Enter the text that you want to speak!")
        self.input_text.grid(row=3, column=0, sticky="we")

        self.synthetize_button = tk.Button(self.wrapper, text="Synthetize", command=self.synthetize_text)
        self.synthetize_button.grid(row=4, column=0, sticky="we")

        
        #Mel spectro
        if self.spectro_widget==None:
            print("Create spectro widget")
            self.spectro_widget=tk.Canvas(self.wrapper)
            self.spectro_widget.grid(column=0,row=5,sticky="nwe")
            self.spectro_widget.create_text(50,50, text="After synth here the mel spectro appears", anchor="nw", fill="black")

        #Synth audio
        self.synth_frame=tk.Frame(self.wrapper)
        self.synth_frame.grid(column=0,row=6,sticky="nwe")
        self.synth_label=tk.Label(self.synth_frame,text="Synthetyzed file name")
        self.synth_label.grid(column=0,row=0,sticky="we")
        self.play_synth_button=tk.Button(self.synth_frame,text="Play",command=lambda v=self.synth_file_name:self.play_sound(v))
        self.play_synth_button.grid(column=1,row=0,sticky="we")


        


    def synthetize_text(self):
        input_text = self.input_text.get("1.0",tk.END)
        filename= self.filename_entry.get()
        filename= None if len(filename)==0 else filename

        self.ckpt_path=self.synth_model_entry.get()
        self.ckpt_step=int(self.synth_ckpt_entry.get())

        if not os.path.exists(self.ckpt_path):
            messagebox.showerror("Error",self.ckpt_path+" does not exists")
            return
        if not os.path.exists(os.path.join(self.ckpt_path,"ckpt","LibriTTS",str(self.ckpt_step)+".pth.tar")):
            messagebox.showerror("Error",os.path.join(self.ckpt_path,"ckpt","LibriTTS",str(self.ckpt_step)+".pth.tar")+" does not exists")
            return
        e_ctrl,p_ctrl,d_ctrl=self.energy_entry.get(),self.pitch_entry.get(),self.duration_entry.get()
        
        print(filename)
        speaker_emb,path=self.get_zero_shot_synth(text=input_text,FILE_PATH=self.browse_folder_path,file_name=filename,
                                 e_ctrl=e_ctrl,p_ctrl=p_ctrl,d_ctrl=d_ctrl)


        
        #Mel spectro
        self.spectro_widget.delete("all") 
        self.spectro_image= Image.open(path+".png")
        self.spectro_image=self.spectro_image.resize((self.spectro_widget.winfo_width(),self.spectro_widget.winfo_height()))
        self.spectro_image=ImageTk.PhotoImage(self.spectro_image)
        self.spectro_widget.create_image(0,0,image=self.spectro_image,anchor="nw")

        #Synth player
        self.synth_file_name=path+".wav"
        print(self.synth_file_name)
        self.synth_label=tk.Label(self.synth_frame,text=self.synth_file_name)
        self.synth_label.grid(column=0,row=0,sticky="nwe")
        self.play_synth_button=tk.Button(self.synth_frame,text="Play",command=lambda v=self.synth_file_name:self.play_sound(v))
        self.play_synth_button.grid(column=1,row=0,sticky="nwe")



   
        return


    def browse_folder(self,ask=True):
        self.items=[]
        self.browse_frame.destroy()
        self.populate_content_list(blank_figure=False)
        if ask:
        
            self.browse_folder_path=filedialog.askdirectory()
        

        file_paths=[os.path.join(self.browse_folder_path,f) for f in os.listdir(self.browse_folder_path) if ".wav" in f]

        print("Current folder:",self.browse_folder_path)
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            var = tk.IntVar()
            c=tk.Checkbutton(self.list_frame, text=f"{i}) {file_name}",variable=var).grid(row=i+1, column=1, sticky="w")
            b=tk.Button(self.list_frame, text="Play", command=lambda file_path=file_path: self.play_sound(file_path))
            b.grid(row=i+1, column=0)
            self.items.append((var,file_path))
            #b.pack()

    def play_sound(self, file_path):
        try:
            playsound.playsound(file_path)
        except Exception as ex:
            messagebox.showerror("Error",ex)

    def display_embeddings(self):
        selected_items=[(i[1],n) for n,i in enumerate(self.items) if i[0].get()==1]
        if len(selected_items)<4:
            messagebox.showerror("Error","You need to select at least 4 file to compute umap.")
            return

        embeddings=[]
        with torch.no_grad():
            for f in selected_items:
                emb=get_speaker_emb(f[0],self.speaker_model).cpu().numpy()
                embeddings.append(emb)
        try:
            embeddings=np.vstack(embeddings)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
            embeddings = reducer.fit_transform(embeddings)
            
            self.embeddings_fig, ax = plt.subplots()
            self.embeddings_fig.figsize=(1, 3)
            self.embeddings_fig.suptitle("Selected speaker embeddings umap")
            ax.scatter(embeddings[:,0], embeddings[:,1], c="red")
            for i, txt in enumerate(selected_items):
                annotation=txt[1]
                ax.annotate(annotation, (embeddings[i,0], embeddings[i,1]))
            
            #width=self.figure_widget.winfo_width()
            self._clear(self.figure_widget)
            self.figure_widget = FigureCanvasTkAgg(self.embeddings_fig, self.content_list)
            self.figure_widget.get_tk_widget().grid(row=2, column=0)
            #self.figure_widget.get_tk_widget().config(width=width)
            #self.figure_widget.draw()
           
        except Exception as ex:
            messagebox.showerror("Error",ex)

    def _clear(self,figure_canvas):
        for item in figure_canvas.get_tk_widget().find_all():
            figure_canvas.get_tk_widget().delete(item)

    def compute_embeddings(self):
    
        selected_files= [item[1] for item in self.items if item[0].get() == 1]#[item[1] for item in items if item[0].get() == 1]
        embs=[]
        with torch.no_grad():
            if selected_files:
                for f in selected_files:
                    emb=get_speaker_emb(f,speaker_model)
                    embs.append(emb)
            else:
                messagebox.showerror("Error", "No items selected.")

        return torch.stack(embs).mean(axis=0)

    def get_zero_shot_synth(self,p_ctrl=1.0,e_ctrl=1.0,d_ctrl=1.0,
                            text="Text example",
                            FILE_PATH="./wavs_example",
                            file_name=None):
        speaker_emb=self.compute_embeddings()

        try:
            p_ctrl=float(p_ctrl)
            e_ctrl=float(e_ctrl)
            d_ctrl=float(d_ctrl)
        except Exception as ex:
            messagebox.showerror("Control converting error",ex)

        args=Args(
            {
            "prepro":f"{self.ckpt_path}/configs/LibriTTS/preprocess.yaml",
            "train":f"{self.ckpt_path}/configs/LibriTTS/train.yaml",
            "model":f"{self.ckpt_path}/configs/LibriTTS/model.yaml",
            "mode":"single",
            "text":text,
            "speaker_input":speaker_emb,
            "restore_step":self.ckpt_step,
            "pitch_control":p_ctrl,
            "energy_control":e_ctrl,
            "duration_control":d_ctrl,
            
            }
        )

        try:
            # Read Config
            preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
            model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
            train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

            preprocess_config["path"]={key:os.path.join("ZeroShotFastSpeech2",path) for key,path in preprocess_config["path"].items()}
            train_config["path"]={key:os.path.join("ZeroShotFastSpeech2",path) for key,path in train_config["path"].items()}
            model_config["vocoder"]["config_path"]=os.path.join("ZeroShotFastSpeech2",model_config["vocoder"]["config_path"])

            configs = (preprocess_config, model_config, train_config)


            # Get model
            model = get_synth_model(args, configs, device, train=False)

            # Load vocoder
            vocoder = get_vocoder(model_config, device)

            ids = raw_texts = [args.text[:100]]
            speakers = args.speaker_input
            print("Speaker embedding shape:",speakers.shape)
            if preprocess_config["preprocessing"]["text"]["language"] == "en":
                texts = np.array([preprocess_english(args.text, preprocess_config)])
            text_lens = np.array([len(texts[0])])
            #
            batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

            control_values = args.pitch_control, args.energy_control, args.duration_control

            if file_name==None:
                ts = int(time.time())
                basename=f"{ts}_cloned"
            else:
                basename=file_name

            i=1
            while os.path.exists(os.path.join(FILE_PATH, "{}.wav".format(basename))):
                if "_copy_" in basename:
                    basename=basename.split("_copy_")[0]
                basename+=f"_copy_{i}"
                i+=1
    
            synthesize(model, args.restore_step, configs, vocoder, batchs, control_values,result_path=FILE_PATH,basenames=[basename])
            
            self.browse_folder(ask=False)
            return speaker_emb,os.path.join(FILE_PATH,basename)
        except Exception as ex:
            messagebox.showerror("Error",str(ex)+"\nRemember to use a different file names if a wav file with this name already exists in browse folder.")




if __name__ == "__main__":
    root = tk.Tk(screenName="Data driven: AI voice cloning")
    #Create a fullscreen window
    #root.attributes('-fullscreen', True)
    root.title("Data driven: AI voice cloning")
    root.state('zoomed')
    speaker_model=get_speaker_model("ecapa","cuda")
    app = SoundApp(speaker_model=speaker_model,master=root)
    app.mainloop()
    exit()




