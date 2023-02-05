import os
import argparse

from preprocess import build_OpenSet_dataset,build_dictionary

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="./data",type=str)
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--dataset",default="openSet",type=str)
    args = parser.parse_args()
    return args
def check_args(args):
    data_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    assert os.path.exists(data_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
def main():
    # build dictionary
    args = get_args()
    check_args(args)
    data_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    load_dict_file = os.path.join(result_dir,"dictionary.txt")
    assert args.dataset in ["openSet"]
    if not os.path.exists(load_dict_file):
        build_dictionary(data_dir,result_dir)
    if args.dataset == "openSet":
        build_OpenSet_dataset(data_dir,result_dir)
    
if __name__ == "__main__":
    main()
