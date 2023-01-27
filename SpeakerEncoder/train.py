from pathlib import Path
from comet_ml import Experiment
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from model.wavlm_speaker_encoder import WavLMSpeakerEncoder
from model.ecapatdnn_speaker_encoder import EcapaTDNNSpeakerEncoder

import os
import umap
import matplotlib.pyplot as plt
import torch_audiomentations as am
from typing import Union

from constants import colormap



def get_model(model_name,device,activation_function,train_only_head,enlarge_head=False):
    model=None
    if ("wavlm" in model_name) and not ("untrained" in model_name):
        pretrained_name="microsoft/"+model_name
        model = WavLMSpeakerEncoder(device,pretrained_name=pretrained_name,activation_function=activation_function)
    elif ("wavlm" in model_name) and ("untrained" in model_name):
        pretrained_name="microsoft/"+model_name
        model = WavLMSpeakerEncoder(device,pretrained_name=pretrained_name,activation_function=activation_function)
     

    elif model_name=="ecapa":
        model = EcapaTDNNSpeakerEncoder(device,activation_function=activation_function)
        for p in model.parameters():
            p.requires_grad=True
    elif model_name=="ecapa-untrained":
        model = EcapaTDNNSpeakerEncoder(device,pretrained_name=None,activation_function=activation_function)
        for p in model.parameters():
            p.requires_grad=True


    if model_name=="wavlm-large" and enlarge_head==True:
        model.model.projector=torch.nn.Linear(in_features=1024,out_features=512,bias=True).to(device)
        model.model.tdnn[0].kernel=torch.nn.Linear(in_features=2560,out_features=512,bias=True).to(device)
        model.model.tdnn[1].kernel=torch.nn.Linear(in_features=1536,out_features=512,bias=True).to(device)
        model.model.tdnn[2].kernel=torch.nn.Linear(in_features=1536,out_features=512,bias=True).to(device)
        model.model.tdnn[3].kernel=torch.nn.Linear(in_features=512,out_features=512,bias=True).to(device)
        model.model.tdnn[4].kernel=torch.nn.Linear(in_features=512,out_features=1500,bias=True).to(device)

        model.model.feature_extractor=torch.nn.Linear(in_features=3000,out_features=1024,bias=True).to(device)
        model.model.classifier=torch.nn.Linear(in_features=1024,out_features=1024,bias=True).to(device)
    #Freeze all weight except for head
    if (model_name in ["wavlm-base","wavlm-large","wavlm-base-plus","wavlm-base-plus-sv","wavlm-base-sv"]) and train_only_head:
        #Frozen all
        for p in model.model.parameters():
            p.requires_grad=False
        #Unfrozen only head
        for p in model.model.projector.parameters():
            p.requires_grad=True
        for p in model.model.tdnn.parameters():
            p.requires_grad=True
        for p in model.model.feature_extractor.parameters():
            p.requires_grad=True

        

    return model


def save_umap(step,model_dir,embeds,speakers,speakers_per_batch,
                utterances_per_speaker,file_name="",multi_rec_session=False,
                annote_point=False):
    
    print("Drawing and saving projections (step %d)" % step)
    file_name=file_name+f"umap_{step:06d}.png"
    projection_fpath = model_dir / file_name

    max_speakers = min(speakers_per_batch, len(colormap))
    embeds = embeds[:(max_speakers) * utterances_per_speaker]

    reducer = umap.UMAP(n_neighbors=(utterances_per_speaker*speakers_per_batch)-1,random_state=42)
    projected = reducer.fit_transform(embeds.cpu().detach().numpy()).reshape((max_speakers,utterances_per_speaker,2))

    sp_label=list(speakers.keys())
    for i in range(speakers_per_batch):
        plt.scatter(projected[i,:,0],projected[i,:,1],color=colormap[i],label=sp_label[i],alpha=0.75)  
        if annote_point:
            for j,u in enumerate(speakers[sp_label[i]]):   
                x=None
                if multi_rec_session:
                    x=str(u.path).split("/")[-1:]#"_".join(str(u.path).split("/")[-2:])
                else:
                    x=u.path.name
                plt.annotate(x,(projected[i,j,0]+0.015,projected[i,j,1]+0.015),size=6,alpha=0.7)    

    lgd=plt.legend(bbox_to_anchor=(1.8, 1), loc='upper right', ncol=2)
    # Create the directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) 
    plt.savefig(projection_fpath,bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    return projection_fpath

def log_audio_batch(experiment,step,speakers,audios,srs,multi_rec_session=False):

    sp_label=list(speakers.keys())
    idx=0
    for i,s in enumerate(sp_label):
        for j,u in enumerate(speakers[sp_label[i]]):
            x=None
            if multi_rec_session:
                x="_".join(str(u.path).split("\\")[-2:])
            else:
                x=u.path.name

            metadata={"step": step,"speaker":s,"utterance":x}
            name=f"step: {step}, speaker: {s}, utterance: {x}"
            experiment.log_audio(audios[idx],sample_rate=srs[idx],metadata=metadata,file_name=name)
            idx+=1

def get_augmentor(noises_paths,reverbs_paths,musics_paths,model_sample_rate):
    pipe=[]
    if musics_paths:
           pipe.append(am.AddBackgroundNoise(p=0.6,background_paths=musics_paths,min_snr_in_db=10,max_snr_in_db=15,sample_rate=16000,target_rate=model_sample_rate))
    if noises_paths:
        pipe.append(am.AddBackgroundNoise(p=0.6,background_paths=noises_paths,min_snr_in_db=10,max_snr_in_db=15,sample_rate=16000,target_rate=model_sample_rate))
    if reverbs_paths:
           pipe.append(am.ApplyImpulseResponse(p=0.6,ir_paths=reverbs_paths,compensate_for_propagation_delay=True,sample_rate=16000,target_rate=model_sample_rate))

    augmentor=am.Compose(pipe)
    return augmentor

def validation_step(loader,model,model_dir,epoch,val_max_epochs,num_step,train_step,experiment,device,
                    speakers_per_batch,utterances_per_speaker,num_samples,
                    vis_val_art_every,multi_rec_session,data_parallel=None,annote_point=False):
    val_loss=[]
    val_eer=[]
    with torch.no_grad():
        for step, speaker_batch in enumerate(loader,start=1):
            
            inputs=speaker_batch.data.to(device)
            embeds=model(inputs)
            print("Forward pass")
            if data_parallel!=None:
                loss, eer = model.module.loss(embeds,speakers_per_batch,utterances_per_speaker)
            else:
                loss, eer = model.loss(embeds,speakers_per_batch,utterances_per_speaker)
            print("Compute Loss")
            val_loss.append(loss)
            val_eer.append(eer)

           
            num_step+=1
            experiment.log_metric(name="loss",value=loss.item(),step=num_step,epoch=epoch)
            experiment.log_metric(name="EER",value=eer,step=num_step,epoch=epoch)

            if vis_val_art_every!=0 and num_step%vis_val_art_every==0: 
                speakers=speaker_batch.files
                image_path=save_umap(embeds=embeds,model_dir=model_dir,step=num_step,speakers=speakers,
                                    speakers_per_batch=speakers_per_batch,annote_point=annote_point,
                                    utterances_per_speaker=utterances_per_speaker,multi_rec_session=multi_rec_session)
                experiment.log_image(image_path,step=num_step, 
                                    name=f"Validation Current loss: {loss.item()}")

                print(f"Pushing val audio to comet")
                log_audio_batch(experiment,num_step,speakers,inputs.detach().cpu().numpy(),speaker_batch.sample_rates,multi_rec_session)
            print("DEBUG: speaker batch file len",len(speaker_batch.files),"total file", sum([len(f) for f in speaker_batch.files.values()]))
           
            print(f"Samples analyzed: {step*speakers_per_batch*utterances_per_speaker}/{num_samples*val_max_epochs}")
            if (step*speakers_per_batch*utterances_per_speaker)>=(num_samples*val_max_epochs):
                print("Validation finished")
                break

    avg_loss=sum(val_loss)/len(val_loss)
    avg_eer=sum(val_eer)/len(val_eer)
    print("\nAvg val loss:",avg_loss)
    print("Avg val eer:",avg_eer,"\n")
    experiment.log_metric(name="avg_loss",value=avg_loss,step=train_step,epoch=epoch)
    experiment.log_metric(name="avg_EER",value=avg_eer,step=train_step,epoch=epoch)      
    return avg_loss,avg_eer,num_step      

def save_model(model,step,epoch,data_parallel,optimizer,scheduler,path):
    real_model=model
    if data_parallel:
        real_model=model.module
    torch.save({
        "step": step + 1,
        "epoch": epoch,
        "model_state": real_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict()
    }, path)

def train_with_comet_viz(
                        model_name:str,run_id: str, 
                        train_data_root: Path, models_dir: Path, 
                        vis_train_art_every: int,vis_train_metrics_every: int,
                        val_every:Union[int,str],vis_val_art_every:int,
                        save_every: int,backup_every: int,  
                        force_restart: bool, comet_key: str,comet_project:str,


                       
                        window_size_s:int,

                        
                        scheduler_factor:float,scheduler_patience:int,weight_decay:float,
                        
                        device: str,num_workers: int,
                        speakers_per_batch:int,utterances_per_speaker:int,
                        val_speakers_per_batch:int=None,val_utterances_per_speaker:int=None,
                        val_max_epochs:int=1,
                        activation_function=None,
                        
                        multi_rec_session=False,
                        learning_rate_init:float=0.0,
                        train_only_head:bool=False,

                        val_data_root:Path=None,


                        model_sample_rate:int=16000,
                        num_epochs:int=None,
                        aug_device:str="best",
                        use_data_augmentation:bool=False,
                        augment_noises_paths:Union[list,str]="./noises",
                        augment_reverbs_paths:Union[list,str]="./reverbs",
                        augment_musics_paths:Union[list,str]="./musics",

                        data_parallel=[0,1],

                        annote_point:bool=False,

                        scheduler_metrics:str="train_loss",
                        scheduler_metrics_avg_every:int=None,
                        enlarge_head=False,

                        accumalate_gradient_every_scheduler=False
                        ):

    print("Setting comet viz...")
    experiment=Experiment(api_key=comet_key,project_name=comet_project)
    experiment.set_name(run_id)
   
    #Create a dataset and a dataloader
    print("Instantiate train dataset/dataloader...")
    dataset = SpeakerVerificationDataset(train_data_root,speakers_per_batch,
                                         utterances_per_speaker,multi_rec_sessions=multi_rec_session,
                                         model_sample_rate=model_sample_rate)
    loader = SpeakerVerificationDataLoader(dataset,num_workers=num_workers,window_size_s=window_size_s)

    #Check for validation and eventually instantiati dataset and dataloader
    dataset_val=None
    loader_val=None
    if val_data_root!=None:
        print("Instantiate val dataset/dataloader...")
        dataset_val = SpeakerVerificationDataset(val_data_root,val_speakers_per_batch,
                                            val_utterances_per_speaker,multi_rec_sessions=multi_rec_session,
                                            model_sample_rate=model_sample_rate)
        loader_val = SpeakerVerificationDataLoader(dataset_val,num_workers=num_workers,window_size_s=window_size_s)
        #Log validation parameters on comet
        log_params={"val_speakers_per_batch":val_speakers_per_batch,
                    "val_utterances_per_speaker":val_utterances_per_speaker,
                    "val_num_samples":dataset_val.n_samples,
                    "val_speakers":len(dataset_val.speakers),
                    "val_step_per_epoch":dataset_val.n_samples//(val_speakers_per_batch*val_utterances_per_speaker)}
        experiment.log_parameters(log_params)
        
    #Check for augemntaion pipe and eventually prepare it
    augmentor=None
    if use_data_augmentation:
        augmentor=get_augmentor(noises_paths=augment_noises_paths,musics_paths=augment_musics_paths,
                                reverbs_paths=augment_reverbs_paths,model_sample_rate=model_sample_rate)
        experiment.log_text(str(augmentor))

    #Prepare device (can be different for augmentation and forward)   
    if device=="best":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if aug_device=="best":
        aug_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nDevice:",device)
    print("Augmentation device:",aug_device)



    # Configure file path for the model
    model_dir = models_dir / run_id    #Store all models of the run and umap
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt" #Based on save_every parameter
    best_fpath=model_dir /"best_model.pt" #Best model based on validation eer and (if equal) loos

    # Create the model and the optimizer
    model=get_model(model_name,device,activation_function,train_only_head=train_only_head,enlarge_head=enlarge_head)
    experiment.set_model_graph(model, overwrite=False)
    optimizer = None
    init_step = 1
    epoch=1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init,weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)
    num_val_step=0
     # Load any existing model
    if not force_restart:
        if state_fpath.exists():         
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            epoch=checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state"]),
            optimizer.load_state_dict(checkpoint["optimizer_state"])

            if learning_rate_init==0.0: #Attention, for restore lr from learning rate set it to 0.0
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            else:
                optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
                # Create the model and the optimizer
        print("Starting the training from scratch.")

    model.train()
    if data_parallel!=None:
        model = torch.nn.DataParallel(model, device_ids=data_parallel)
        print("Model sended to data parallel")



    log_params={
        "model_name":model_name,
        "learning_rate_init":optimizer.param_groups[0]["lr"],
        "speakers_in_dataset":len(dataset.speakers),
        "speakers_per_batch":speakers_per_batch,
        "utterances_per_speaker":utterances_per_speaker,
        "sim_activation_function":str(activation_function.__name__) if activation_function is not None else str(None),
        "device":str(torch.cuda.get_device_name(0) if device=="cuda" else "CPU"),
        "scheduler_patience":scheduler_patience,
        "scheduler_factor":scheduler_factor,
        "weight_decay":weight_decay,
        "train_only_head":train_only_head,
        "dataset_path":str(train_data_root),
        "dataset_samples":dataset.n_samples,
        "dataset_step_per_epoch":dataset.n_samples//(speakers_per_batch*utterances_per_speaker),
        "window_size_seconds":window_size_s,
        "use_augmentation":use_data_augmentation,
        "scheduler_metrics":scheduler_metrics,
        "scheduler_metrics_avg_every":scheduler_metrics_avg_every,
        "accumulation_gradient_scheduler_logic":accumalate_gradient_every_scheduler,
        "enlarge_head":enlarge_head,
        "data_parallel":data_parallel,
        "annote_point":annote_point
                }
    experiment.log_parameters(log_params)
    
    

    # Training loop
    last_loss=None
    avg_val_loss=avg_val_eer=float("inf")
    step_per_epoch=dataset.n_samples//(speakers_per_batch*utterances_per_speaker)

    #Configure scheduler metrics updating
    scheduler_losses=[]
    scheduler_eers=[]
    scheduler_avg_loss=None
    scheduler_avg_eer=None  
    if "val" in scheduler_metrics and  scheduler_metrics_avg_every!=None:
        scheduler_metrics_avg_every=None
        print(f"ATTENTION: scheduler updating is based on validation metrics, the step of the scheduler is in synch with the val_every params")


    optimizer.zero_grad()
    try: #Catch exception to end comet experiment 
        for step, speaker_batch in enumerate(loader, init_step):
        
            inputs = speaker_batch.data
            if augmentor!=None:
                inputs=augmentor(speaker_batch.data.to(aug_device).unsqueeze(1),sample_rate=model_sample_rate).squeeze(1) #Aggiungi una dimensione per il channel (mono), poi lo rimuovo
                print("Audio augmented")

           
            # Forward pass
            inputs=inputs.to(device)
            embeds=model(inputs)
            print("Forward pass")
   
            loss,eer=None,None
            if data_parallel:
                loss,eer=model.module.loss(embeds,speakers_per_batch,utterances_per_speaker)         
                loss.backward()
                print("Compute Loss done")
                model.module.do_gradient_ops()
                print("Backward pass done")
            else:
                loss, eer = model.loss(embeds,speakers_per_batch,utterances_per_speaker)
                 # Backward pass
                loss.backward()
                print("Compute Loss done")
                model.do_gradient_ops()
                print("Backward pass done")

            if accumalate_gradient_every_scheduler==False:
                optimizer.step()
                optimizer.zero_grad()
                print("Parameters update")

            scheduler_losses.append(loss)
            scheduler_eers.append(eer)  
       
            if scheduler_metrics_avg_every!=None and step%scheduler_metrics_avg_every==0:
                if accumalate_gradient_every_scheduler==True:
                    optimizer.step()
                    optimizer.zero_grad()
                    print("Parameters update, every train step scheduler logic")
                scheduler_avg_loss=sum(scheduler_losses)/len(scheduler_losses)
                scheduler_avg_eer=sum(scheduler_eers)/len(scheduler_eers)
                scheduler_eers=[]
                scheduler_losses=[]
                experiment.log_metric(name=f"loss_scheduler_avg{scheduler_metrics_avg_every}",value=scheduler_avg_loss,step=step,epoch=epoch)
                experiment.log_metric(name=f"eer_scheduler_avg{scheduler_metrics_avg_every}",value=scheduler_avg_eer,step=step,epoch=epoch)

                #Use this value for scheduler (otherwise can use validation eer or loss)
                if "val" not in scheduler_metrics:
                    if "eer" in scheduler_metrics:
                        scheduler.step(scheduler_avg_eer)
                    elif "loss" in scheduler_metrics:
                        scheduler.step(scheduler_avg_loss)
              
      

           

            # Update visualizations
            speakers=speaker_batch.files
            with experiment.context_manager("train"):
                if step%vis_train_metrics_every==0:
                   
                   
                     
                    experiment.log_metric(name="loss",value=loss.item(),step=step,epoch=epoch)
                    experiment.log_metric(name="EER",value=eer,step=step,epoch=epoch)
                    experiment.log_metric(name="lr",value=optimizer.param_groups[0]["lr"],step=step,epoch=epoch)
               

            
                # Store umap and audio file to the backup folder
                if vis_train_art_every != 0 and step % vis_train_art_every == 0:
                    image_path=save_umap(embeds=embeds,model_dir=model_dir,step=step,speakers=speakers,
                                speakers_per_batch=speakers_per_batch,annote_point=annote_point,
                                utterances_per_speaker=utterances_per_speaker,multi_rec_session=multi_rec_session)
                    experiment.log_image(image_path,step=step, 
                                        name=f"step: {step} Current loss: {loss.item()} Last loss: {last_loss}")

                    print(f"Pushing audio to comet step({step})")
                    log_audio_batch(experiment,step,speakers,inputs.detach().cpu().numpy(),speaker_batch.sample_rates,multi_rec_session)
                
                #Restore umap and audio in case of exploding loss (new loss >= 5*last loss)
                if last_loss!=None and loss.item()>=last_loss*5 and not (vis_train_art_every != 0 and step % vis_train_art_every == 0):
                    print(f"Step({step}) exploding loss")
                
                
                    image_path=save_umap(embeds=embeds,model_dir=model_dir,step=step,speakers=speakers,
                                speakers_per_batch=speakers_per_batch,
                                utterances_per_speaker=utterances_per_speaker,annote_point=annote_point,
                                file_name="exploding_loss_",multi_rec_session=multi_rec_session)
                    experiment.log_image(image_path,step=step,
                                        name=f"step: {step} Current loss: {loss.item()} Last loss: {last_loss} Exploding loss")
                    
                    print(f"Pushing audio to comet step({step})")
                    log_audio_batch(experiment,step,speakers,inputs.detach().cpu().numpy(),speaker_batch.sample_rates,multi_rec_session)
                
            #Validation step eventually save best model
            with experiment.context_manager("validation"):
                if step%val_every==0 and val_data_root!=None:
                    print(f"Validation step start (val every {val_every} step)")
                    new_avg_val_loss,new_avg_val_eer,num_val_step=validation_step(loader_val,model,model_dir,epoch,val_max_epochs,
                                    num_val_step,step,experiment,
                                    device,val_speakers_per_batch,val_utterances_per_speaker,dataset_val.n_samples,
                                    vis_val_art_every,multi_rec_session,data_parallel=data_parallel,annote_point=annote_point)
                    
                    if "val" in scheduler_metrics:
                        if accumalate_gradient_every_scheduler==True:
                            optimizer.step()
                            optimizer.zero_grad()
                            print("Parameters update, val scheduler logic")
                        if "eer" in scheduler_metrics:
                            scheduler.step(new_avg_val_eer)
                        elif "loss" in scheduler_metrics:
                            scheduler.step(new_avg_val_loss)
                         

                    if (new_avg_val_eer<avg_val_eer) or ((new_avg_val_eer==avg_val_eer) and (avg_val_loss>new_avg_val_loss)):
                
                        print(f"FOUNDED NEW BEST MODEL old eer: {avg_val_eer} new eer: {new_avg_val_eer} old loss: {avg_val_loss} new loss: {new_avg_val_loss} saving it! ")
                        avg_val_eer=new_avg_val_eer
                        avg_val_loss=new_avg_val_loss
                        print("Saving the model (step %d)" % step)
                        save_model(model,step,epoch,data_parallel,
                                    optimizer,scheduler,best_fpath)


            # Overwrite the latest version of the model
            if save_every != 0 and step % save_every == 0:
                print("Saving the model (step %d)" % step)
                save_model(model,step,epoch,data_parallel,
                                    optimizer,scheduler,state_fpath)

            # Make a backup
            if backup_every != 0 and step % backup_every == 0:
                print("Making a backup (step %d)" % step)
                backup_fpath = model_dir / f"encoder_{step:06d}.bak"
                save_model(model,step,epoch,data_parallel,
                                    optimizer,scheduler,backup_fpath)
               

            last_loss=loss.item()
            print(f"\nStep {step} finished------------\n")
            if num_epochs!=None and epoch==num_epochs:
                print("Number of epochs reached, training finished.")
                experiment.end()
                return
            if step%step_per_epoch==0:
                epoch+=1
        print("Train loop out!")

    except KeyboardInterrupt as ex:
        print("Interrupted:","KeybordInterruption")
        experiment.end()
    except IndexError as ex:
        print("Interrupted:",ex)
        experiment.end()
    except Exception as ex:
        print("Interrupted:",ex)
        experiment.end()