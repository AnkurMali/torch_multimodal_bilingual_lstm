import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import PIL
import math
from bpemb import BPEmb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.io.image import read_image
from torchvision.models import ResNet50_Weights
import torchvision

class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, maxlen=64, transform=None):
        self.data = df
        self.tokenizer = tokenizer
        if transform == None:
            self.transform = self._transform()
        else:
            self.transform = transform
        self.maxlen = maxlen
        
    def __getitem__(self, index):
        text = self.data.caption.values[index]
        image_path = self.data.image_path[index]
        
        if not text.endswith('.'):
            text = text + '.'
    
        encoded = self.tokenizer.encode(text, maxlen=self.maxlen)
        
        input_ids = encoded[:-1]
        label_ids = encoded[1:]
        im = read_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        # image = im.convert('RGB')
        image = self.transform(im)

        return {'input_ids': torch.tensor(input_ids), 
                'label_ids':torch.tensor(label_ids),
                'image': image}
    
    def __len__(self):
        return len(self.data)
    
    def _transform(self):
#         transform = torchvision.transforms.Compose([
#             # Resize image to 224 x 224 as required by most vision models
#             torchvision.transforms.Resize(
#                 size=(224, 224)
#             ),
#             # Convert PIL image to tensor with image values in [0, 1]
#             torchvision.transforms.ToTensor(),

#             torchvision.transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
        weights = ResNet50_Weights.DEFAULT
        transform=weights.transforms()
        return transform