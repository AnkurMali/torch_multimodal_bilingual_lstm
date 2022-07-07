import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from bpemb import BPEmb
from torch import nn

class MultiBPE:
    def __init__(self, vs=320000, dim=300, add_pad_emb=True):
        self.multibpemb = BPEmb(lang="multi", vs=vs, dim=dim, add_pad_emb=add_pad_emb)
        self.k2i = self.multibpemb.emb.key_to_index
        self.i2k = self.multibpemb.emb.index_to_key
        self._embedding_matrix = self.multibpemb.emb.vectors

    def encode(self, 
               text, 
               padding=True,
               use_eos=True,
               maxlen=64):
        if use_eos:
            ids = self.multibpemb.encode_ids_with_bos_eos(text)
        else:
            ids = self.multibpemb.encode_ids(text)
        
        seq_len = len(ids)
        
        if maxlen != None:
            if seq_len > maxlen:
                ids = ids[:maxlen]  

            if padding:
                pad_seq = [self.k2i['<pad>']] * (maxlen-seq_len)
                ids.extend(pad_seq)
        return ids
    
    def decode(self, ids):
        if 320000 in ids:
            id_pad = ids.index(320000)
            ids = ids[:id_pad]
        return self.multibpemb.decode_ids(ids)

    def decode_with_pad(self, ids):
        ret = []
        
        for id_ in ids:
            ret.append(self.i2k[id_])
        
        return ' '.join(ret)
        
    def get_embedding_matrix(self):
        return self._embedding_matrix