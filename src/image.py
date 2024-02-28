import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
#from utils import readLatLong, transformSteps

def load_data(img_path,train = True, inference=False):
    gt_path = img_path.replace('.jpg','.h5')#.replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    # new line here
    #npimg = np.array(img)
    #img = cv2.resize(npimg,(npimg.shape[1]*2,npimg.shape[0]*2), interpolation = cv2.INTER_AREA)
    #img = Image.fromarray(img)
    #print(npimg.max())
    #print(npimg.min())

    if train == False:
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

    #print(np.sum(target),np.sum(target1), np.sum(target2),np.sum(target3))
    #target = cv2.resize(target,(int(target.shape[1]/4),int(target.shape[0]/4)), interpolation = cv2.INTER_AREA)*64

    return img,target3


def load_inference(chip_path):
    chip = Image.open(chip_path).convert('RGB')
    return chip
