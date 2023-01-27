from train import train_with_comet_viz

from pathlib import Path
import json
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', help='Path of the json config file')


    args=parser.parse_args()

    fp=open(args.config_path,"r")
    config=json.load(fp)
    fp.close()

    print("Config:\n",config)

    config["models_dir"]=Path(config["model_name"])
    config["train_data_root"]=Path(config["train_data_root"])
    if config["val_data_root"]!=None:
        config["val_data_root"]=Path(config["val_data_root"])

    train_with_comet_viz(**config)