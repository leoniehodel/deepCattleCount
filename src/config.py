import time

import torch
import torch.nn as nn
import torch.optim as optim


class TrainingConfig:
    def __init__(self, args):
        self.original_lr = 1e-5
        self.lr = 1e-5
        self.batch_size = 16
        self.momentum = 0.95
        self.decay = 5*1e-4
        self.start_epoch = 0
        self.epochs = 80
        self.steps = [40]
        self.scales = [1]
        self.workers = 4
        self.seed = time.time()
        self.print_freq = 30
        # the training tiles are 420 x 420 px on disk and are resized to this size, the scale
        # the v2 model was trained at; inference.py feeds its 420 px chips at the same size
        self.img_size = (424, 424)
        self.counter = 0
        self.pre = args.pre
        self.task = args.task
        self.use_cuda = torch.cuda.is_available()
        self.best_prec1 = 1e6

    def get_criterion(self):
        return nn.MSELoss(reduction='sum').cuda() if self.use_cuda else nn.MSELoss(reduction='sum')

    def get_optimizer(self, parameters):
        return optim.Adam(parameters, self.lr, weight_decay=self.decay)

