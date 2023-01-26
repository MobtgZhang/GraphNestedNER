import torch
import torch.nn as nn

class VGAE(nn.Module):
    def __init__(self):
        super(VGAE,self).__init__()
    def forward(self,):
        pass
class GraphNestedModel(nn.Module):
    def __init__(self,config):
        super(GraphNestedModel,self).__init__()
        self.emb_dim = config.emb_dim
        self.num_chars = config.num_chars
        self.type_class = config.type_class

        self.char_emb = nn.Embedding(config.num_chars,config.emb_dim)
        self.mask_emb = nn.Embedding(2,config.emb_dim)
        
        if hasattr(config,"type-class"):
            self.type_class = config.type_class
    def forward(self,sent,mask):
        char_emb = self.char_emb(sent)
        mask_emb = self.mask_emb(mask)
        vemb = char_emb*self.gelu(mask_emb)

