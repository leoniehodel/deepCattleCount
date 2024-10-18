import random

import cv2
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class listDataset(Dataset):
    def __init__(
            self, root, shape=None, shuffle=True, transform=None,  train=False,
            seen=0, batch_size=1, num_workers=4
        ):
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

def load_data(img_path,train = True, inference=False):
    gt_path = img_path.replace('.jpg','.h5')#.replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    if not train:
        crop_size = (int(img.size[0]/2),int(img.size[1]/2))
        if random.randint(0,9)<= -1:
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        # GAME(L)?
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        # data augmentation
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # due to network architecture target is 8 times smaller than input
    # value is value*64
    target1 = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)), interpolation = cv2.INTER_AREA)*64
    target2 = cv2.resize(target1, (55,55),interpolation = cv2.INTER_AREA)
    target3 = target2 * (target1.shape[0]/float(55) * target1.shape[1]/float(55))

    return img,target3


def load_inference(chip_path):
    chip = Image.fromarray(chip_path).convert('RGB')
    return chip