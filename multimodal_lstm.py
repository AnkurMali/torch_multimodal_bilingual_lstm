import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import torchvision
import PIL
import math
from bpemb import BPEmb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#LOAD_PATH = './data/'

seed = 420

torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

class MultiModalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim=1, bias=True, is_multimodal=True):
        super(MultiModalLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        if torch.cuda.is_available():
            self.lstm = nn.LSTMCell(input_dim, hidden_dim).cuda()
        else:
            self.lstm = nn.LSTMCell(input_dim, hidden_dim)
            
        self.is_multimodal = is_multimodal
        
        if self.is_multimodal:
            if torch.cuda.is_available():
                # M = Variable(torch.nn.init.uniform_(
                #     torch.empty((img_context.size(1), self.hidden_dim),  dtype=torch.float), 
                #     a = -0.1, 
                #     b = 0.1).cuda())
                self.M = torch.nn.Linear(2048, self.hidden_dim, bias = False).cuda()
            else:
                # M = Variable(torch.nn.init.uniform_(
                #     torch.empty((img_context.size(1), self.hidden_dim),  dtype=torch.float), 
                #     a = -0.1, 
                #     b = 0.1))
                self.M = torch.nn.Linear(2048, self.hidden_dim, bias = False)
    
    def forward(self, x, img_context):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            hn = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            hn = Variable(torch.zeros(x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            cn = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            cn = Variable(torch.zeros( x.size(0), self.hidden_dim))
        
        outs = []
        for seq in range(x.size(1)):
            if self.is_multimodal:
                d = self.M(img_context)
                hn = hn * d
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            outs.append(hn.clone())
            
    
        if torch.cuda.is_available():
            out = torch.stack(outs).to('cuda')
        else:
            out = torch.stack(outs)
        out = out.permute((1,0,2))
        return out, (hn, cn)
    
    def forward_text(self, x):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            hn = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            hn = Variable(torch.zeros(x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            cn = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            cn = Variable(torch.zeros( x.size(0), self.hidden_dim))
        
        outs = []
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            outs.append(hn.clone())
            
    
        if torch.cuda.is_available():
            out = torch.stack(outs).to('cuda')
        else:
            out = torch.stack(outs)
        out = out.permute((1,0,2))
        return out, (hn, cn)