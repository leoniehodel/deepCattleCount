import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from src.image import *
from torchvision import transforms
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        if shuffle:
            random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img,target = load_data(img_path,self.train)

        if self.transform is not None:
            img = self.transform(img)
        return img,target

class InferDataset(Dataset):
    def __init__(self, root, transform=None, batch_size = 1, num_workers=4):
        #super(InferDataset, self,).__init__()
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img = load_inference(img_path)
        return self.transform(img)
