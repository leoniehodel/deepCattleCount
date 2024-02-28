import os
import argparse
import torch
import torch.nn as nn
import json
import geopandas as gp
import pandas as pd
import time
import glob
import numpy as np
from pyproj import Transformer
from src.model import CSRNet
from src.utils import cutpatches, readLatLong, findImgDims
from src import dataset
from torch.autograd import Variable
from torchvision import transforms

parser = argparse.ArgumentParser(description='Inference CSRNet')

parser.add_argument('inference_json', metavar='INFERENCE',type=str,
                    help='path to json file with dir/ of target images')
parser.add_argument('modelparameters', metavar='MODPARS',type=str,
                    help='path to folder with parameters of the ensemble (.tar files)')


def main():
    global args
    args = parser.parse_args()
    args.seed = time.time()

    try:
        modelparameters = args.modelparameters
    except AttributeError:
        modelparameters = False

    args.img_size = (440, 440)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed(int(args.seed))
    print(modelparameters)
    dirs = os.listdir(modelparameters)
    model_list = [(i) for i in dirs if i.endswith(".tar")]


    args.batch_size = 16

    # all folders to imgs to do inference on
    with open(args.inference_json, 'r') as outfile:
        inference_list = json.load(outfile)

    for satimg in range(len(inference_list)):

        # just add a for loop for every model
        img_path = inference_list[satimg]
        print('doing inference on ...', img_path)

        cutpatches(img_path)
        chips_list = []
        list = os.path.join(img_path, 'chips/')

        for chip in glob.glob(os.path.join(list, '*.jpg')):
            chips_list.append(chip)

        chips_list.sort()

        # read the georeference
        xdim, ydim = findImgDims(chips_list)
        tab_path = [_ for _ in os.listdir(img_path) if _.endswith('.tab')]

        latlong = readLatLong(os.path.join(img_path, tab_path[0]))

        transformer = Transformer.from_crs("epsg:4326", "epsg:4674")
        x1, x2 = transformer.transform(latlong[0], latlong[1])
        y1, y2 = transformer.transform(latlong[2], latlong[3])
        latlong = (x1, x2, y1, y2)
        res1 = (latlong[1] - latlong[0]) / xdim
        res2 = (latlong[3] - latlong[2]) / ydim
        # transform1 = Affine.translation(latlong[0], latlong[2]) * Affine.scale(res1, res2)

        transformer.transform(12, 12)
        # what i want is an array with the latlongs
        coordy = np.zeros(ydim)  # 6
        coordx = np.zeros(xdim)  # 12

        for x in range(xdim):
            coordx[x] = latlong[0] + res1 * x + (res1 / 2)
        for y in range(ydim):
            coordy[y] = latlong[2] + res2 * y + (res2 / 2)

        chips_content = np.zeros((len(model_list), xdim*ydim), dtype=np.float64)
        # print(chips_list)

        n_model = 0
        for i in model_list:
            model = CSRNet()
            model = model.cuda()
            print(modelparameters)
            checkpoint = torch.load(os.path.join(modelparameters, i))
            model.load_state_dict(checkpoint['state_dict'])
	    # load it into the inference loader
            chips_content[n_model, :] = inference(chips_list, model)  

            print('sum of ',i,' : ',np.sum(chips_content[n_model, :]))

            n_model  =n_model +1


        df = pd.DataFrame({
            'n_cattle': np.mean(np.array(chips_content),axis=0),
            'n_cattle_sd': np.std(chips_content,axis=0),
            'Longitude': np.repeat(coordx, ydim),  # repeat entire array ydim times 12*6
            'Latitude': np.tile(coordy, xdim)  # repeat single elements xdim times
        })
        df2 = pd.DataFrame({
            'n_cattle0': chips_content[0,:],
            'n_cattle1': chips_content[1,:],
            'n_cattle2': chips_content[2,:],
            'n_cattle3': chips_content[3,:],
            'n_cattle4': chips_content[4,:],
            'Longitude': np.repeat(coordx, ydim),  # repeat entire array ydim times 12*6
            'Latitude': np.tile(coordy, xdim)  # repeat single elements xdim times
        })


        gpd = gp.GeoDataFrame(df, crs="EPSG:4674", geometry=gp.points_from_xy(df.Longitude, df.Latitude))
        name = img_path.split("/")[-3] + "_" + img_path.split("/")[-2] + "_" + img_path.split("/")[-1] + ".geojson"
        # the name has to be [MUN_UF_#image]
        print('saving the geojson ... ', name)
        with open('data/inference/' + name, 'w') as file:
            file.write(gpd.to_json())


def inference(chips_path, model):
    infer_data = dataset.InferDataset(chips_path, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.img_size)]))
    infer_loader = torch.utils.data.DataLoader(infer_data, batch_size=args.batch_size)

    cattle_count = np.zeros(len(chips_path))
    batch = 0

    with torch.no_grad():
        for img in infer_loader:
            img = img.cuda()
            output = model(img)
            output = output.data.sum((2, 3)).cpu().squeeze(0).detach().numpy()

            cattle_count[(batch * args.batch_size):((batch + 1) * args.batch_size)] = np.round(output.squeeze(1))
            batch += 1
    # print(cattle_count)
    return cattle_count

if __name__ == '__main__':
    main()
