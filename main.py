import os
import argparse

from torch.utils.data import DataLoader

from src.data import Dictionary,EntitiesDataset

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
    args = get_args()
    check_args(args)
    result_dir = os.path.join(args.result_dir,args.dataset)
    train_dataset = EntitiesDataset(result_dir,"train")

    for k in range(len(train_dataset)):
        print(train_dataset[k])
    exit()
    dev_file = os.path.join(result_dir,"dev.json")
    dev_dataset = EntitiesDataset(dev_file,data_dict)
    test_file = os.path.join(result_dir,"test.json")
    test_dataset = EntitiesDataset(test_file,data_dict)


if __name__ == "__main__":
    main()
