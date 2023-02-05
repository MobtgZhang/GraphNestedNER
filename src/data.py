import os
import json

import numpy as np

from torch.utils.data import Dataset

def batch_fy(batch):
    max_length = max([len(bt["sentence"]) for bt in batch])
    sent_idxs = [bt["sentence"]+(max_length-len(bt["sentence"]))[0] for bt in batch]
    sent_mask = [len(bt["sentence"])*[1]+(max_length-len(bt["sentence"]))[0] for bt in batch]
    return 

class EntitiesDataset(Dataset):
    def __init__(self,result_dir,tag_name):
        super(EntitiesDataset,self).__init__()
        load_file = os.path.join(result_dir,"%s.json"%tag_name)
        dict_file = os.path.join(result_dir,"dictionary.txt")
        data_dict = Dictionary.load(dict_file)
        label_file = os.path.join(result_dir,"labels.txt")
        self.idx2label = {}
        self.label2idx = {}
        with open(label_file,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                index,word = line.strip().split("\t")
                self.idx2label[int(index)] = word
                self.label2idx[word] = int(index)
        self.data_set = []
        with open(load_file,mode="r",encoding="utf-8") as rfp:
            for item in rfp:
                item = json.loads(item)
                for tp_item in item["labeled entities"]:
                    tp_item[2] = self.label2idx[tp_item[2]]
                item["sentence"] = [data_dict[word] for word in item["sentence"]]
                self.data_set.append(item)
    def __getitem__(self, index):
        sentence = self.data_set[index]["sentence"]
        tagged_mat = np.zeros(shape=(len(sentence),len(sentence)),dtype=np.int64)
        for item in self.data_set[index]["labeled entities"]:
            tagged_mat[item[0],item[1]] = item[2]
        return sentence,tagged_mat
    def __len__(self):
        return len(self.data_set)

class Dictionary:
    def __init__(self,UNK="UNKTOKEN",PAD="PADTOKEN"):
        self.words2id = {PAD:0,UNK:1}
        self.id2words = [PAD,UNK]
        self.start_id = -1
    def add(self,word):
        if word not in self.words2id:
            self.words2id[word] = len(self.words2id)
            self.id2words.append(word)
    def __getitem__(self,key):
        if type(key) == str:
            if key == " ":
                return self.words2id[0]
            return self.words2id.get(key,1)
        elif type(key) == int:
            return self.id2words[key]
        else:
            raise TypeError("The key type %s is unknown."%str(type(key)))
    def __next__(self):
        if self.start_id>=len(self.id2words)-1:
            self.start_id = -1
            raise StopIteration()
        else:
            self.start_id += 1
            return self.id2words[self.start_id]
    def __iter__(self):
        return self
    def __repr__(self):
        re_str = "Dictionary (%d)"%len(self.id2words)
        return re_str
    def __str__(self):
        re_str = "Dictionary (%d)"%len(self.id2words)
        return re_str
    def __len__(self):
        return len(self.id2words)
    @staticmethod
    def load(load_file):
        tmp_dict = Dictionary()
        words2id = {}
        id2words = []
        with open(load_file,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                idx,key = line.strip().split("\t")
                id2words.append(key)
                words2id[key] = int(idx)
        tmp_dict.id2words = id2words
        tmp_dict.words2id = words2id
        return tmp_dict
    def save(self,save_file):
        with open(save_file,mode="w",encoding="utf-8") as wfp:
            for idx,key in enumerate(self.id2words):
                write_line = "%d\t%s\n"%(idx,key)
                wfp.write(write_line)

