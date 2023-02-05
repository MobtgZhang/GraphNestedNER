import os
import argparse

import yaml

class Config:
    def __init__(self,config_file):
        with open(config_file,mode="r",encoding="utf-8") as rfp:
            config = yaml.load(rfp,Loader=yaml.FullLoader)
        for key in config:
            if not hasattr(self,key):
                setattr(self,key,config[key])
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--config-dir",default="./config",type=str)
    parser.add_argument("--dataset",default="openSet",type=str)
    parser.add_argument("--model",default="GraphNestedModel",type=str)
    parser.add_argument("--batch-size",default=12,type=int)
    parser.add_argument("--epoches",default=10,type=int)
    parser.add_argument("--learning-rate",default=1e-2,type=float)
    
    args = parser.parse_args()
    return args
def check_args(args):
    data_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    config_file = os.path.join(args.config_dir,args.model+".yaml")
    assert os.path.exists(data_dir)
    assert os.path.exists(config_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
def get_config(args):
    config_file = os.path.join(args.config_dir,args.model+".yaml")
    config = Config(config_file)
    return config
    
