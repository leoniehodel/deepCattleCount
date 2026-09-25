import argparse
import os

import torch
import torch.nn as nn

from src.config import TrainingConfig
from src.dataset import find_samples
from src.model import CSRNet
from src.training import adjust_learning_rate, train, validate
from src.utils import save_checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CSRNet')
    parser.add_argument('--train_folder', metavar='TRAIN', required=True,
                        help='folder with the training images and their .h5 density maps')
    parser.add_argument('--test_folder', metavar='TEST', required=True,
                        help='folder with the test images and their .h5 density maps')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,help='path to the pretrained model')
    parser.add_argument('--dilation', type=int, default=3,
                        help='dilation rate of the backend: 3 (default, as the v2 model), '
                             '2 as the v1 ensemble. Has to match the weights given with --pre')
    parser.add_argument('gpu', metavar='GPU', type=str, help='id of the GPU to train on, e.g. 0')
    parser.add_argument('task',metavar='TASK', type=str,help='task id to use.')
    return parser.parse_args()

def main():

    args = parse_arguments()
    # make the chosen GPU the only one torch sees, so every .cuda() call lands on it. This
    # has to happen before torch first queries CUDA, which TrainingConfig does
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = TrainingConfig(args)
    best_prec1 = config.best_prec1
   
    train_list = find_samples(args.train_folder)
    val_list = find_samples(args.test_folder)
    print(f'{len(train_list)} training and {len(val_list)} test images')

    torch.cuda.manual_seed(int(config.seed))
    model = CSRNet(dilation_val=args.dilation)
    model = model.cuda() if config.use_cuda else model

    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        config.lr,
        weight_decay=config.decay
    )

    if config.pre:
        if os.path.isfile(config.pre):
            print("=> loading checkpoint '{}'".format(config.pre))
            checkpoint = torch.load(config.pre)
            config.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.pre))
            
    for epoch in range(config.start_epoch, config.epochs):
        adjust_learning_rate(optimizer, epoch, config)
        train(train_list, model, criterion, optimizer, epoch, config)
        prec1 = validate(val_list, model, config)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.task)


if __name__ == '__main__':
    main()        
