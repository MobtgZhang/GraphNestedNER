import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MutiAttentionLayer(nn.Module):
    def __init__(self,dim_q,dim_k,dim_v):
        super(MutiAttentionLayer, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q_mat = nn.Linear(dim_q,dim_v)
        self.k_mat = nn.Linear(dim_k,dim_v)
        self.v_mat = nn.Linear(dim_v,dim_v)
        self._norm_fact = 1.0/math.sqrt(dim_k)

    def forward(self,query,key=None,value=None):
        if key is None:
            assert self.dim_k == self.dim_q
            key = query
        if value is None:
            assert self.dim_k == self.dim_v
            value = key
        q_mat = self.q_mat(query) # (batch_size,seq_lenk,dim_k)
        k_mat = self.k_mat(key) # (batch_size,seq_lenq,dim_k)
        v_mat = self.v_mat(value) # (batch_size,seq_lenv,dim_v)
        atten = torch.bmm(q_mat,k_mat.permute(0,2,1))*self._norm_fact # (batch_size,seq_lenq,seq_lenk)
        atten = F.softmax(atten,dim=-1)
        o_mat = torch.bmm(atten,v_mat) # (batch_size,seq_lenq,dim_v)
        return o_mat,atten


class VGAE(nn.Module):
    def __init__(self):
        super(VGAE,self).__init__()
    def forward(self,):
        pass
class GraphNestedModel(nn.Module):
    def __init__(self,config):
        super(GraphNestedModel,self).__init__()
        self.emb_dim = config.emb_dim
        self.hid_dim = config.hid_dim
        self.num_chars = config.num_chars

        self.char_emb = nn.Embedding(config.num_chars,config.emb_dim)
        self.mask_emb = nn.Embedding(2,config.emb_dim)

        self.linA = nn.Linear(config.emb_dim,config.hid_dim)
        self.linB = nn.Linear(config.emb_dim,config.hid_dim)
        self.gelu = nn.GELU()

        
        if hasattr(config,"label_class"): 
            self.label_class = config.label_class
            self.split = nn.Linear(config.emb_dim,self.label_class)
    def forward(self,sent,mask):
        char_emb = self.char_emb(sent)
        mask_emb = self.mask_emb(mask)
        vemb = char_emb*self.gelu(mask_emb)

        # make an attention for the model
        va = self.linA(vemb)
        vb = torch.sigmoid(self.linB(vemb))
        vc = 1.0/self.hid_dim*torch.matmul(va,vb.transpose(1,2))
        if hasattr(self,"label_class"): 
            vd = torch.relu(self.split(vemb))
            vd = vd.unsqueeze(2)
            vc = vc.unsqueeze(3)
            pred = torch.matmul(vc,vd)
            pred = F.softmax(pred,dim=-1)
        else:
            pred = torch.sigmoid(vc)
        return pred

