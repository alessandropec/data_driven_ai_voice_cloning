import os
from tqdm import tqdm
import argparse



def verify_pairs_list(pairs_list_path,data_root_path,new_pairs_list_path,error_list_path):
    fpw=open(new_pairs_list_path,"w")
    fpwe=open(error_list_path,"w")
    with open(pairs_list_path,"r") as fp:
        lines=fp.readlines()
        for l in tqdm(lines):
            ls=l.split(" ")
            p1,p2=os.path.join(data_root_path,ls[1].strip()),os.path.join(data_root_path,ls[2].strip())
      
            if os.path.exists(p1) and os.path.exists(p2):
                fpw.write(l)
                continue
            elif not os.path.exists(p1) or not os.path.exists(p2):
                fpwe.write(l)
                continue

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--pairs_list_path")
    parser.add_argument("--data_root_path")
    parser.add_argument("--new_pairs_list_path")
    parser.add_argument("--error_list_path")

    args=parser.parse_args()

    verify_pairs_list(**vars(args))