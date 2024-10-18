import csv
import os
import shutil

import h5py
import numpy as np
import torch
from PIL import Image


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')


def readLatLong(tabfile):
    tmp = []
    with open(tabfile, 'r', newline='') as coord:
        coord_reader = csv.reader(coord, delimiter='\t')
        for game in coord_reader:
            tmp.append(game)
        lat1 = float(tmp[7][0].split(' ')[2].split('(')[1].split(',')[0])
        lat2 = float(tmp[8][0].split(' ')[2].split('(')[1].split(',')[0])
        long1 = float(tmp[7][0].split(' ')[3].split(')')[0])
        long2 = float(tmp[8][0].split(' ')[3].split(')')[0])
   
    return (lat1, lat2, long1, long2)


def findmaxXPatch(listdir):
    all_xes = [ listdir[i].split('_')[3].split('-')[0] for i in range(0,len(listdir))]
    list1 = [int(x) for x in all_xes]
    list1.sort()
    return list1[-1]

def findXPatch(name):
    number = name.split('_')[3].split('-')[0]
    return int(number)

def findYPatch(name):
    number = name.split('_')[3].split('-')[1].split('.')[0]
    return int(number)

def cutpatches(img_dir):
    patch_dir = os.path.join(img_dir,'patches/')
    chip_dir = os.path.join(img_dir,'chips/')
    # Check whether the output path exists or not
    isExist = os.path.exists(chip_dir)    #cur_x = 1

    if not isExist:
        os.makedirs(chip_dir)
    else:
        return
    # make a statement here

    listdir = os.listdir(patch_dir)

    listdir.sort()
    division = 6

    for img in listdir:
        if img.endswith('.jpg'):

            #open the image as jpg
            im = Image.open(patch_dir+img)
            #convert into np array
            nparr = np.array(im)
            # split into [division] pieces on the x axis
            subnparr = np.array_split(nparr,division, axis =1)

            # chips name should be counted though the whole image and not
            # thorugh a single patch
            date = img.split('_')
            name = date[0]+'_'+date[1]+'_'+date[2]

            yPatchnumber = findYPatch(img)
            xPatchnumber = findXPatch(img)
            xChipnumber = (xPatchnumber-1)*6+1
            yChipnumber = (yPatchnumber-1)*6+1

            for subarr in range(division):
                # split into [division] pieces on y axis ()
                tmp = np.array_split(subnparr[subarr],division, axis=0)


                for subsub in range(division):
                    chip = Image.fromarray(tmp[subsub])
                    if(xChipnumber<100 and yChipnumber<100):
                        if (xChipnumber<10 and yChipnumber<10):
                            chip.save(chip_dir+name+'_'+'0'+'0'+str(xChipnumber)+'_'+'0'+'0'+str(yChipnumber)+'.jpg')
                        elif(yChipnumber<10):
                            chip.save(chip_dir+name+'_'+'0'+str(xChipnumber)+'_'+'0'+'0'+str(yChipnumber)+'.jpg')
                        elif(xChipnumber<10):
                            chip.save(chip_dir+name+'_'+'0'+'0'+str(xChipnumber)+'_'+'0'+str(yChipnumber)+'.jpg')
                        else:
                            chip.save(chip_dir+name+'_'+'0'+str(xChipnumber)+'_'+'0'+str(yChipnumber)+'.jpg')

                    elif(yChipnumber<100 and xChipnumber>99):
                        if (yChipnumber<10):
                                chip.save(chip_dir+name+'_'+str(xChipnumber)+'_'+'0'+'0'+str(yChipnumber)+'.jpg')
                        else:
                            chip.save(chip_dir+name+'_'+str(xChipnumber)+'_'+'0'+str(yChipnumber)+'.jpg')
                    elif(xChipnumber<100 and yChipnumber>99):
                        if (xChipnumber<10):
                            chip.save(chip_dir+name+'_'+'0'+'0'+str(xChipnumber)+'_'+str(yChipnumber)+'.jpg')
                        else:
                            chip.save(chip_dir+name+'_'+'0'+str(xChipnumber)+'_'+str(yChipnumber)+'.jpg')
                    else:
                        chip.save(chip_dir+name+'_'+str(xChipnumber)+'_'+str(yChipnumber)+'.jpg')

                    yChipnumber +=1
                #after 6 times through y, increse x
                xChipnumber  += 1
                yChipnumber = (yPatchnumber-1)*6+1



def findImgDims(sorted_chip_list):

    #print(sorted_chip_list)
    ydim = int(sorted_chip_list[-1].split('/')[-1].split('.')[0].split('_')[4])
    #ydime = sorted_chip_list[-1].split('/')[-1].split('_')[4]
    print('ydim is ', ydim)
    xdim = int(sorted_chip_list[-1].split('/')[-1].split('.')[0].split('_')[3])
    print('xdim is ', xdim)

    return xdim, ydim
