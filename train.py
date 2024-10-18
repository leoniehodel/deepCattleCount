import argparse
import json
import os

import torch
import torch.nn as nn

from src.config import TrainingConfig
from src.model import CSRNet
from src.training import adjust_learning_rate, train, validate
from src.utils import save_checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CSRNet')
    parser.add_argument('--train_json', metavar='TRAIN',help='path to train json')
    parser.add_argument('--test_json', metavar='TEST', help='path to test json')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,help='path to the pretrained model')
    parser.add_argument('gpu',metavar='GPU', type=str, help='GPU id to use.')
    parser.add_argument('task',metavar='TASK', type=str,help='task id to use.')
    return parser.parse_args()

def main():

    args = parse_arguments()
    config = TrainingConfig(args)
    best_prec1 = config.best_prec1
   
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(int(config.seed))
    model = CSRNet()
    model = model.cuda() if config.use_cuda else model

    criterion = nn.MSELoss(size_average=False).cuda()
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
