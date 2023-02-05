import os
import json
import shutil

from src.data import Dictionary

def build_OpenSet_dataset(data_dir,result_dir):
    load_tags_list = ["train","dev","test"]
    save_tags_list = ["train","valid","test"]
    label_set = set()
    for load_tag,save_tag in zip(load_tags_list,save_tags_list):
        load_file = os.path.join(data_dir,"%s.json"%load_tag)
        save_file = os.path.join(result_dir,"%s.json"%save_tag)
        with open(load_file,mode="r",encoding="utf-8") as rfp:
            data_list = json.load(rfp)
            with open(save_file,mode="w",encoding="utf-8") as wfp:            
                for item in data_list:
                    item["sentence"] = eval(item["sentence"])
                    item["labeled entities"] = eval(item["labeled entities"])
                    for tp_item in item["labeled entities"]:
                        label_set.add(tp_item[2])
                    wfp.write(json.dumps(item,ensure_ascii=False)+"\n")
    save_classes_file = os.path.join(result_dir,"labels.txt")
    label_set.add("UNK")
    with open(save_classes_file,mode="w",encoding="utf-8") as wfp: 
        for idx,word in enumerate(label_set):
            wfp.write(str(idx)+"\t"+word+"\n")
    
def build_dictionary(data_dir,result_dir,test=False):
    tags_list = ["train","dev"]
    save_file = os.path.join(result_dir,"dictionary.txt")
    if test:
        tags_list += ["test"]
    data_dict = Dictionary()
    for tag_name in tags_list:
        load_file = os.path.join(data_dir,tag_name+".json")
        with open(load_file,mode="r",encoding="utf-8") as rfp:
            data_list = json.load(rfp)
            for item in data_list:
                sentence = item["sentence"]
                for word in sentence:
                    word = word.strip()
                    if word!="":
                        data_dict.add(word)
    data_dict.save(save_file)
