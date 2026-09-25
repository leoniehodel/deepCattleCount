import os
import random

import cv2
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def find_samples(folder):
    # every image in the folder that has a density map (.h5) of the same name next to it
    samples, missing = [], []
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXTENSIONS:
            continue
        if os.path.isfile(os.path.join(folder, stem + '.h5')):
            samples.append(os.path.join(folder, name))
        else:
            missing.append(name)
    if missing:
        print(f'skipping {len(missing)} image(s) in {folder} without a .h5 density map: '
              + ', '.join(missing[:5]) + (' ...' if len(missing) > 5 else ''))
    if not samples:
        raise FileNotFoundError(f'no image with a matching .h5 density map found in {folder}')
    return samples


class listDataset(Dataset):
    def __init__(
            self, root, shape=None, shuffle=True, transform=None,  train=False,
            seen=0, batch_size=1, num_workers=4, chip_size=(424, 424)
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
        self.chip_size = chip_size

        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        img,target = load_data(img_path,self.train, chip_size=self.chip_size)

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

def load_data(img_path,train = True, chip_size=(424, 424)):
    gt_path = os.path.splitext(img_path)[0] + '.h5'
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # data augmentation: random left-right flip of the training images, no cropping
    if train and random.random() > 0.8:
        target = np.fliplr(target)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # due to network architecture target is 8 times smaller than input
    # value is value*64
    target1 = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)), interpolation = cv2.INTER_AREA)*64
    # and it has to match the size of the output, which is 1/8 of the size the image is resized to
    out_h, out_w = chip_size[0] // 8, chip_size[1] // 8
    target2 = cv2.resize(target1, (out_w, out_h),interpolation = cv2.INTER_AREA)
    target3 = target2 * (target1.shape[0]/float(out_h) * target1.shape[1]/float(out_w))

    return img,target3


def load_inference(chip_path):
    chip = Image.fromarray(chip_path).convert('RGB')
    return chip