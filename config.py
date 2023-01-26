import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--result-dir",default="./result",type=str)
    args = parser.parse_args()
    return args
def check_args(args):
    assert os.path.exists(args.data_dir)
    
