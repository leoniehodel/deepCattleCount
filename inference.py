import os
import argparse
import torch
import geopandas as gp
import pandas as pd
import time
import numpy as np
from src.model import CSRNet
from src import dataset
from torchvision import transforms
from xml.etree import ElementTree as ET
from pyproj import CRS
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser(description='Inference CSRNet')

parser.add_argument('modelparameters', metavar='MODPARS',type=str,
                    help='path to folder with parameters of the ensemble (.tar files)')

parser.add_argument('path_to_img', metavar='IMG',type=str,
                    help='path to image to be processed')

parser.add_argument('path_to_kml', metavar='KML',type=str,
                    help='path to kml georeference of the image')
def main():
    global args
    args = parser.parse_args()
    args.seed = time.time()

    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed(int(args.seed))


    # read in ensemble of model parameters
    try:
        modelparameters = args.modelparameters
    except AttributeError:
        modelparameters = False

    dirs = os.listdir(modelparameters)
    model_list = [(i) for i in dirs if i.endswith(".tar")]
    args.batch_size = 16
    args.img_size = (420, 420)
    # image preprocessing: cut the image into chips to analyze individually
    img_path = args.path_to_img
    print('doing inference on ...', img_path)

    # cut the image into single chips
    desired_chip_size = 420
    img = Image.open(os.path.join(img_path))
    np_img = np.array(img)
    img_height, img_width = np_img.shape[:2]

    # Calculate the number of chips in each dimension
    xdim = img_width // desired_chip_size
    ydim = img_height // desired_chip_size

    # Adjust x_dim and y_dim
    if img_width % desired_chip_size != 0:
        xdim += 1
    if img_height % desired_chip_size != 0:
        ydim += 1
    # Initialize a list to store the chips
    chips_img_list = []

    # crop image
    for x in range(xdim):
        for y in range(ydim):
            left = x * desired_chip_size
            upper = y * desired_chip_size
            right = left + desired_chip_size
            lower = upper + desired_chip_size

            chip = img.crop((left, upper, right, lower))
            chips_img_list.append(np.array(chip))

    # Initialize a list to store the model outputs
    chips_content = np.zeros((len(model_list), xdim*ydim), dtype=np.float64)

    # ensemble prediction on image chips
    n_model = 0
    for i in model_list:
        model = CSRNet()
        model = model.cuda()
        checkpoint = torch.load(os.path.join(modelparameters, i))
        model.load_state_dict(checkpoint['state_dict'])
        # load it into the inference loader
        chips_content[n_model, :] = inference(chips_img_list, model)
        print('sum of ',i,' : ',np.sum(chips_content[n_model, :]))
        n_model  =n_model +1

    # get geospatial information from kml file
    kml_path = args.path_to_kml
    latlong = read_latlong_kml(kml_path)
    input_crs = CRS.from_epsg(4326)
    # west east south north
    res1 = (latlong[1] - latlong[0]) / xdim # east - west
    res2 = (latlong[3] - latlong[2]) / ydim # north -south

    # Generate latitude and longitude arrays
    coordx = np.zeros(xdim)
    coordy = np.zeros(ydim)

    for x in range(xdim):
        # west + x_step
        coordx[x] = latlong[0] + res1 * x + (res1 / 2)
    for y in range(ydim):
        # south + y_steps
        coordy[y] = latlong[2] + res2 * y + (res2 / 2)

    # Reshape, flip, and transform the content arrays
    mean_content = np.mean(chips_content, axis=0).reshape(xdim, ydim)
    std_content = np.std(chips_content, axis=0).reshape(xdim, ydim)
    mean_content_flipped = np.fliplr(mean_content)
    std_content_flipped = np.fliplr(std_content)

    # Create DataFrame
    df = pd.DataFrame({
        'n_cattle': mean_content_flipped.ravel(),  # Flatten the array
        'n_cattle_sd': std_content_flipped.ravel(),  # Flatten the array
        'Longitude': np.repeat(coordx, ydim),  # Repeat entire array ydim times
        'Latitude': np.tile(coordy, xdim) }) # Repeat single elements xdim times

    gpd = gp.GeoDataFrame(df, crs=input_crs, geometry=gp.points_from_xy(df.Longitude, df.Latitude))
    geojson_path = os.path.splitext(img_path)[0] + ".geojson"

    print('saving the geojson ... ')

    # Write the GeoDataFrame to a JSON file
    with open(geojson_path, 'w') as file:
        file.write(gpd.to_json())

    print(f"File saved to {geojson_path}")


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

def read_latlong_kml(kml_file):
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Extract coordinates from the LatLonBox tag
    north = float(root.find('.//north').text)
    south = float(root.find('.//south').text)
    east = float(root.find('.//east').text)
    west = float(root.find('.//west').text)

    return west, east, south, north


if __name__ == '__main__':
    main()
