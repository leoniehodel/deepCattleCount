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
        self.img_size = (440, 440)
        self.counter = 0
        self.pre = args.pre
        self.task = args.task
        self.use_cuda = torch.cuda.is_available()
        self.best_prec1 = 1e6

    def get_criterion(self):
        return nn.MSELoss(size_average=False).cuda() if self.use_cuda else nn.MSELoss(size_average=False)

    def get_optimizer(self, parameters):
        return optim.Adam(parameters, self.lr, weight_decay=self.decay)
    

class InferenceConfig:
    def __init__(self, args):
        self.modelparameters = args.modelparameters
        self.img_path = args.path_to_img
        self.kml_path = args.path_to_kml
        self.batch_size = 16
        self.img_size = (420, 420)
        self.desired_chip_size = 420
        self.use_cuda = torch.cuda.is_available()
        self.seed = args.seed

    def get_device(self):
        return torch.device('cuda' if self.use_cuda else 'cpu')
