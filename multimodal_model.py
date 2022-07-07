import torch
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
from multimodal_lstm import MultiModalLSTM
from multi_bpe import MultiBPE
from torchvision.models import resnet50, ResNet50_Weights
from multimodal_gpt import GPT2LMHeadModel

multi_bpe = MultiBPE()

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


class MMGPT(nn.Module):
    
    def __init__(self, 
                 num_features=2048, 
                 vocab_size=320001, 
                 n_hidden=64, 
                 n_layers=2, 
                 drop_prob=0.2, 
                 is_multimodal=True,
                 train_visual_module=False):
        super(MMGPT, self).__init__()

        self.drop_prob = drop_prob
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        
        
        self.train_visual_module= train_visual_module
        self.lstm = nn.LSTM(self.num_features, self.n_hidden, batch_first=True)
        ## define the LSTM
        self.gpt = self._load_gpt()
        
        ## define a dropout layer        
        ## define the fully-connected layer        
        self.is_multimodal = is_multimodal
        if is_multimodal:
            self.resnet = self._load_resnet()
    
    def _load_gpt(self):
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        model_id = "gpt2"#"sberbank-ai/mGPT"
        model = GPT2LMHeadModel.from_pretrained(model_id, local_files_only=False).to(device)
        return model
    
    def _load_resnet(self):
        resnet50_org = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet50_processed = torch.nn.Sequential(*list(resnet50_org.children())[:-1])
        if not self.train_visual_module:
            for param in resnet50_processed.parameters():
                param.requires_grad = False
        else:
            print("training resnet parameters...")
        return resnet50_processed
        
    
    
    def forward(self, caption, img, attention_mask=None, labels=None):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        
        context_matrix = self.resnet(img)
        context_matrix = context_matrix.view(context_matrix.size(0),
                                            context_matrix.size(1))
        context_matrix = context_matrix.unsqueeze(1).repeat(1, caption.size(1), 1)
        visual_embedding, _ = self.lstm(context_matrix)
        ## Get the outputs and the new hidden state from the lstm
        gpt_output = self.gpt(input_ids=caption, 
                              visual_context=visual_embedding, 
                              attention_mask=attention_mask, 
                              labels=labels)


        # return the final output and the hidden state
        return gpt_output


class MMLSTM(nn.Module):
    
    def __init__(self, 
                 num_features=300, 
                 vocab_size=320001, 
                 n_hidden=256, 
                 n_layers=2, 
                 drop_prob=0.2, 
                 is_multimodal=True,
                 train_visual_module=False):
        super(MMLSTM, self).__init__()

        self.drop_prob = drop_prob
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        
        
        self.train_visual_module= train_visual_module
        ml_matrix = self._load_multilingual_embedding()
        self.emb_layer = nn.Embedding.from_pretrained(ml_matrix, 
                                                      freeze=False,
                                                      padding_idx=320000)

        ## define the LSTM
        self.lstm = MultiModalLSTM(self.num_features, self.n_hidden, is_multimodal=is_multimodal)
        
        ## define a dropout layer        
        ## define the fully-connected layer
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.n_hidden, self.vocab_size )  
        
        self.is_multimodal = is_multimodal
        if is_multimodal:
            self.resnet = self._load_resnet()
    
    def _load_resnet(self):
        resnet50_org = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet50_processed = torch.nn.Sequential(*list(resnet50_org.children())[:-1])
        if not self.train_visual_module:
            for param in resnet50_processed.parameters():
                param.requires_grad = False
        else:
            print("training resnet parameters...")
        return resnet50_processed
        
    def _load_multilingual_embedding(self):
        matrix = multi_bpe.get_embedding_matrix()
        matrix = torch.tensor(matrix)
        return matrix
    
    
    def forward(self, caption, img):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(caption)     
        
        if self.is_multimodal:
            context_matrix = self.resnet(img)
            context_matrix = context_matrix.view(context_matrix.size(0),
                                                context_matrix.size(1))
        else:
            context_matrix = None
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, _ = self.lstm(embedded, context_matrix)
        
        ## pass through a dropout layer
        lstm_output = self.dropout(lstm_output)
        ## put "out" through the fully-connected layer
        out = self.fc(lstm_output)

        # return the final output and the hidden state
        return out
    
    def forward_text(self, caption):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(caption)     
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, _ = self.lstm.forward_text(embedded)
        
        ## pass through a dropout layer
        lstm_output = self.dropout(lstm_output)
        ## put "out" through the fully-connected layer
        out = self.fc(lstm_output)

        # return the final output and the hidden state
        return out
    
class BenchmarkLSTM(nn.Module):
    
    def __init__(self, num_features=300, vocab_size=320001, n_hidden=256, n_layers=1, drop_prob=0.2, use_xlm=True):
        super(BenchmarkLSTM, self).__init__()

        self.drop_prob = drop_prob
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        
        ml_matrix = self._load_multilingual_embedding()
        self.emb_layer = nn.Embedding.from_pretrained(ml_matrix, 
                                              freeze=False,
                                              padding_idx=320000)

        ## define the LSTM
        self.lstm = nn.LSTM(self.num_features, self.n_hidden, self.n_layers, batch_first=True)
        
        ## define a dropout layer        
        ## define the fully-connected layer
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.n_hidden, self.vocab_size )  
    
    def _load_multilingual_embedding(self):
        matrix = multi_bpe.get_embedding_matrix()
        matrix = torch.tensor(matrix)
        return matrix
    
    def forward(self, x):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, _ = self.lstm(embedded)
        
        ## pass through a dropout layer
        lstm_output = self.dropout(lstm_output)
        ## put "out" through the fully-connected layer
        out = self.fc(lstm_output)

        # return the final output and the hidden state
        return out
    
class BenchmarkCustomLSTM(nn.Module):
    
    def __init__(self, num_features=300, vocab_size=320001, n_hidden=256, n_layers=1, drop_prob=0.2, use_xlm=True):
        super(BenchmarkCustomLSTM, self).__init__()

        self.drop_prob = drop_prob
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        
        ml_matrix = self._load_multilingual_embedding()
        self.emb_layer = nn.Embedding.from_pretrained(ml_matrix, 
                                              freeze=False,
                                              padding_idx=320000)

        ## define the LSTM
        self.lstm = MultiModalLSTM(self.num_features, self.n_hidden, is_multimodal=False)
        
        ## define a dropout layer        
        ## define the fully-connected layer
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.n_hidden, self.vocab_size )  
    
    def _load_multilingual_embedding(self):
        matrix = multi_bpe.get_embedding_matrix()
        matrix = torch.tensor(matrix)
        return matrix
    
    def forward(self, x):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)     
        
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, _ = self.lstm(embedded, None)
        
        ## pass through a dropout layer
        lstm_output = self.dropout(lstm_output)
        ## put "out" through the fully-connected layer
        out = self.fc(lstm_output)

        # return the final output and the hidden state
        return out

