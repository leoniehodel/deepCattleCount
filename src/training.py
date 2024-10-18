import time

import torch
from torchvision import transforms

from src import dataset


def train(train_list, model, criterion, optimizer, epoch, config):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Resize(config.img_size)
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=config.batch_size,
                       num_workers=config.workers),
        batch_size=config.batch_size)
    print(
        'epoch %d of %d, processed %d samples, lr %.10f' % 
        (epoch, config.epochs, epoch * len(train_loader.dataset), config.lr)
    )
    
    model.train()
    end = time.time()
    
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)   
        img = img.cuda()
        pred = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = criterion(pred,target)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        batch_time.update(time.time() - end)
        end = time.time()
        config.counter += config.counter

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validate(val_list, model, config):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Resize(config.img_size)
                   ]),  train=False),
    batch_size=config.batch_size)    
    
    model.eval()
    mae = 0
    batches = 0
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        output= pred
        
        mae += (output.data.sum((2, 3)).cpu() - target.sum((2, 3)).type(torch.FloatTensor)).abs().mean(0)
        batches = batches +1
       
    mae = mae/batches
    print(len(test_loader.dataset))
    print(' * MAE {mae:.3f} '
              .format(mae=mae.item()))


    return mae.item()
        
def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    config.lr = config.original_lr
    
    for i in range(len(config.steps)):
        scale = config.scales[i] if i < len(config.scales) else 1
        if epoch >= config.steps[i]:
            config.lr = config.lr * scale

            if epoch == config.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    