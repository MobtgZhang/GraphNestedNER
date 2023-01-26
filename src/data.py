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
            return self.words2id.get(key,0)
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

