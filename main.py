import os
import argparse
import yaml

import torch

from torch.utils.data import DataLoader

from src.data import Dictionary,EntitiesDataset,batch_fy
from src.model import GraphNestedModel

from config import get_args,get_config,check_args

def main():
    args = get_args()
    check_args(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # dataset preparation
    result_dir = os.path.join(args.result_dir,args.dataset)
    dict_file = os.path.join(result_dir,"dictionary.txt")
    data_dict = Dictionary.load(dict_file)
    train_loader = DataLoader(EntitiesDataset(result_dir,"train",data_dict),batch_size=args.batch_size,shuffle=True,collate_fn=batch_fy)
    valid_loader = DataLoader(EntitiesDataset(result_dir,"valid",data_dict),batch_size=args.batch_size,shuffle=True,collate_fn=batch_fy)
    config = get_config(args)
    config.num_chars = len(data_dict)
    config.label_class = len(train_loader.dataset.label2idx)
    # the model defination
    model = GraphNestedModel(config).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    batch_size = args.batch_size
    label_class = config.label_class

    for epoch in range(args.epoches):
        loss_all = 0.0
        for item in train_loader:
            optimizer.zero_grad()
            sent,mask,label = item
            sent = torch.tensor(sent,dtype=torch.long).to(device)
            mask = torch.tensor(mask,dtype=torch.long).to(device)
            label = torch.tensor(label,dtype=torch.long).to(device)
            predict = model(sent,mask)
            label = label.flatten()
            predict = predict.view(-1,label_class)
            loss = loss_fn(predict,label)
            loss_all += loss.item()
            optimizer.step()
        loss_all /= len(train_loader)
        print("The loss is %0.4f"%loss_all)
if __name__ == "__main__":
    main()
