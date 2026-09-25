import argparse
import os
import sys
import time
import warnings
from xml.etree import ElementTree as ET

import geopandas as gp
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from pyproj import CRS, Geod, Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, from_bounds
from torchvision import transforms

from src import dataset
from src.model import CSRNet

Image.MAX_IMAGE_PIXELS = None

# the image is cut into chips of this size, which are resized to 424 x 424 for the model
CHIP_SIZE = 420

parser = argparse.ArgumentParser(description='Inference CSRNet')

parser.add_argument('modelparameters', metavar='MODPARS',type=str,
                    help='path to folder with parameters of the ensemble (.tar files)')

parser.add_argument('path_to_img', metavar='IMG',type=str,
                    help='path to the image to be processed, 8-bit RGB. A GeoTIFF or JPEG2000 '
                         'carries its own georeference, a JPEG or PNG needs a kml')

parser.add_argument('path_to_kml', metavar='KML',type=str, nargs='?', default=None,
                    help='path to a kml that georeferences the image, for images without their own '
                         'georeference. Takes precedence over the one in the image if given')

parser.add_argument('--output', choices=('counts', 'density-map', 'both'), default='counts',
                    help="what to write: 'counts' one geo-referenced point per chip with the "
                         "predicted number of cattle (default), 'density-map' the stitched "
                         "density surface as a raster and an overlay, or 'both'")

parser.add_argument('--dilation', type=int, default=3,
                    help='dilation rate of the backend, which has to match the weights: '
                         '3 for the 2025 v2 model (default), 2 for the v1 ensemble')


def inference(chips_path, model):
    # returns the density map of every chip, (n_chips, height, width) in head per output cell.
    # CSRNet predicts a surface rather than a number: summing it gives the count, keeping it
    # gives the map
    infer_data = dataset.InferDataset(chips_path, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.img_size)]))
    infer_loader = torch.utils.data.DataLoader(infer_data, batch_size=args.batch_size)

    density = []

    with torch.no_grad():
        for img in infer_loader:
            img = img.to(device)
            output = model(img)
            density.append(output.data[:, 0].cpu().detach().numpy())
    return np.concatenate(density)


def stitch(density, xdim, ydim):
    # one map per chip -> one map over the whole image. The chips were cropped x-major,
    # so chip (x, y) sits at index x * ydim + y
    height, width = density.shape[1:]
    mosaic = np.zeros((ydim * height, xdim * width), dtype=np.float32)
    for x in range(xdim):
        for y in range(ydim):
            mosaic[y * height:(y + 1) * height, x * width:(x + 1) * width] = density[x * ydim + y]
    return mosaic


def read_image(img_path):
    # returns the image as an 8-bit RGB PIL image and its georeference, (affine transform, crs)
    # or None. GeoTIFFs and JPEG2000 files carry a georeference, plain JPEGs and PNGs need a kml
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(img_path) as src:
            # the model is trained on 8-bit RGB tiles, and other data would give wrong counts
            # without any error, e.g. 16-bit values or a near-infrared band read as red
            if src.dtypes[0] != 'uint8' or src.count < 3:
                sys.exit(f'{img_path} has {src.count} band(s) of {src.dtypes[0]}; the model needs '
                         '8-bit RGB imagery (bands 1-3 are read as red, green, blue)')
            georef = (src.transform, src.crs) if src.crs is not None else None
            # JPEGs and PNGs are decoded by Pillow, as in training, since decoders differ slightly
            # and that shifts the counts by around a percent; the rest is read by GDAL
            if src.driver in ('JPEG', 'PNG'):
                pixels = None
            else:
                pixels = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    if pixels is None:
        img = Image.open(img_path)
        if img.mode not in ('RGB', 'RGBA'):
            sys.exit(f'{img_path} is a {img.mode} image; the model needs 8-bit RGB imagery')
        img = img.convert('RGB')
    else:
        img = Image.fromarray(pixels)
    return img, georef


def read_kml(kml_path, width, height):
    # the LatLonBox of a kml ground overlay: the corners of the image in longitude and latitude
    root = ET.parse(kml_path).getroot()
    # a kml usually declares a namespace (xmlns="http://www.opengis.net/kml/2.2"), which
    # ElementTree puts in front of every tag; strip it so the tags can be found by name
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    if root.find('.//LatLonBox') is None:
        sys.exit(f'{kml_path} has no LatLonBox with the corners of the image')
    north = float(root.find('.//north').text)
    south = float(root.find('.//south').text)
    east = float(root.find('.//east').text)
    west = float(root.find('.//west').text)
    rotation = root.find('.//rotation')
    if rotation is not None and float(rotation.text) != 0:
        sys.exit(f'{kml_path} rotates the image; rotated kml overlays are not supported')
    return from_bounds(west, south, east, north, width, height), CRS.from_epsg(4326)


def check_resolution(georef, width, height):
    # the model is trained on imagery of ~30 cm per pixel. At other resolutions the animals
    # appear larger or smaller than in training, which changes the counts without any error
    transform, crs = georef
    to_lonlat = Transformer.from_crs(crs, 4326, always_xy=True)
    centre, right, below = (transform * p for p in
                            ((width / 2, height / 2), (width / 2 + 1, height / 2), (width / 2, height / 2 + 1)))
    lon, lat = to_lonlat.transform(*centre)
    geod = Geod(ellps='WGS84')
    xres = geod.inv(lon, lat, *to_lonlat.transform(*right))[2]
    yres = geod.inv(lon, lat, *to_lonlat.transform(*below))[2]
    print(f'pixel size {xres:.2f} x {yres:.2f} m')
    if not 0.2 <= (xres + yres) / 2 <= 0.4:
        print(f'WARNING: the model is trained on imagery of ~0.3 m per pixel; the counts at '
              f'{(xres + yres) / 2:.2f} m per pixel are unreliable')


def overlay(img, mosaic):
    # the density drawn over the imagery, with the opacity growing with density so that
    # empty ground stays visible and the animals show through the hot colours
    import matplotlib

    dens = np.array(Image.fromarray(mosaic).resize(img.size, Image.BILINEAR))
    positive = dens[dens > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    norm = np.clip(dens / vmax, 0, 1)

    rgb = matplotlib.colormaps['inferno'](norm)[..., :3] * 255
    alpha = (np.clip(norm / 0.45, 0, 1) ** 0.8 * 0.7)[..., None]
    base = np.asarray(img.convert('RGB'), dtype=np.float64)
    return Image.fromarray((base * (1 - alpha) + rgb * alpha).astype(np.uint8))

def write_density_map(density, img, xdim, ydim, georef, img_path):
    # the map covers the chips, which start at the upper-left corner of the image; a cell covers
    # CHIP_SIZE / 53 image pixels
    mosaic = stitch(density, xdim, ydim)
    out_h, out_w = density.shape[1:]
    stem = os.path.splitext(img_path)[0]

    # the density itself, one band of head per cell, at the resolution the model predicts at,
    # as a GeoTIFF in the coordinate reference system of the image
    print('saving the density map ... ')
    profile = dict(driver='GTiff', width=mosaic.shape[1], height=mosaic.shape[0], count=1,
                   dtype='float32', compress='deflate')
    if georef is not None:
        transform, crs = georef
        profile.update(transform=transform * Affine.scale(CHIP_SIZE / out_w, CHIP_SIZE / out_h), crs=crs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(stem + '_density.tif', 'w', **profile) as dst:
            dst.write(mosaic, 1)
    print(f'File saved to {stem + "_density.tif"} '
          f'({mosaic.shape[1]} x {mosaic.shape[0]} cells, {mosaic.sum():.1f} head)')

    # and the same map drawn over the imagery, to look at
    overlay(img.crop((0, 0, xdim * CHIP_SIZE, ydim * CHIP_SIZE)), mosaic).save(stem + '_density.png')
    print(f'File saved to {stem + "_density.png"}')


def main():
    global args
    global device
    args = parser.parse_args()
    args.seed = time.time()

    torch.cuda.manual_seed(int(args.seed))

    # read in ensemble of model parameters
    try:
        modelparameters = args.modelparameters
    except AttributeError:
        modelparameters = False

    dirs = os.listdir(modelparameters)
    model_list = [(i) for i in dirs if i.endswith('.tar')]
    args.batch_size = 16
    # crop 420, feed 424: the training tiles are 420 px on disk and the training pipeline
    # resizes them to 424 before the forward pass, so this is the scale the model was fit to
    args.img_size = (424, 424)
    # image preprocessing: cut the image into chips to analyze individually
    img_path = args.path_to_img
    print('doing inference on ...', img_path)

    img, georef = read_image(img_path)
    img_width, img_height = img.size
    # a kml given on the command line takes precedence over the georeference in the image
    if args.path_to_kml:
        georef = read_kml(args.path_to_kml, img_width, img_height)
    if georef is None:
        if args.output in ('counts', 'both'):
            parser.error(f'--output {args.output} needs a georeferenced image: a GeoTIFF or a kml')
    else:
        check_resolution(georef, img_width, img_height)

    # cut the image into single chips. The pixels at the right and bottom edge that do not fill
    # a whole chip are left out: a chip padded with black gives false counts along the border
    desired_chip_size = CHIP_SIZE
    xdim = img_width // desired_chip_size
    ydim = img_height // desired_chip_size
    if xdim == 0 or ydim == 0:
        sys.exit(f'{img_path} is {img_width} x {img_height} px, smaller than one chip of '
                 f'{CHIP_SIZE} x {CHIP_SIZE} px')
    left_out_x = img_width - xdim * CHIP_SIZE
    left_out_y = img_height - ydim * CHIP_SIZE
    if left_out_x or left_out_y:
        share = 1 - xdim * ydim * CHIP_SIZE ** 2 / (img_width * img_height)
        print(f'leaving out the last {left_out_x} px columns and {left_out_y} px rows, which do not '
              f'fill a {CHIP_SIZE} px chip ({share:.1%} of the image)')

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
    chips_density = []
    # an NVIDIA GPU if there is one, else the GPU of an Apple Silicon Mac, else the CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(device)
    # ensemble prediction on image chips
    n_model = 0
    for i in model_list:
        # Instantiate the model. load_weights=True skips the VGG16 download, since the
        # checkpoint loaded below holds the frontend weights as well
        print(f'loading {i} with dilation {args.dilation}')
        model = CSRNet(load_weights=True, dilation_val=args.dilation)
        # load the weights on the CPU and then move the model to the device
        checkpoint = torch.load(os.path.join(modelparameters, i), map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

        # Load model for inference
        density = inference(chips_img_list, model)
        chips_density.append(density)
        chips_content[n_model, :] = np.round(density.sum(axis=(1, 2)))

        # Print sum of inference results
        print(f'sum of {i} : {np.sum(chips_content[n_model, :])}')

        # Increment model index
        n_model += 1

    if args.output in ('density-map', 'both'):
        write_density_map(np.mean(chips_density, axis=0), img, xdim, ydim, georef, img_path)
    if args.output == 'density-map':
        return

    # one point per chip, at its centre, in longitude and latitude
    transform, crs = georef
    to_lonlat = Transformer.from_crs(crs, 4326, always_xy=True)
    longitude, latitude = [], []
    for x in range(xdim):
        for y in range(ydim):  # the order of the chips
            lon, lat = to_lonlat.transform(*(transform * ((x + 0.5) * CHIP_SIZE, (y + 0.5) * CHIP_SIZE)))
            longitude.append(lon)
            latitude.append(lat)

    df = pd.DataFrame({
        'n_cattle': np.mean(chips_content, axis=0),
        'Longitude': longitude,
        'Latitude': latitude})

    # the standard deviation across the ensemble members is only defined for an ensemble;
    # a single model (e.g. the v2 weights) gives no uncertainty, so the column is left out
    if len(model_list) > 1:
        df.insert(1, 'n_cattle_sd', np.std(chips_content, axis=0))

    gpd = gp.GeoDataFrame(
        df, crs=CRS.from_epsg(4326), geometry=gp.points_from_xy(df.Longitude, df.Latitude)
    )
    geojson_path = os.path.splitext(img_path)[0] + '.geojson'

    print('saving the geojson ... ')

    # Write the GeoDataFrame to a JSON file
    with open(geojson_path, 'w') as file:
        file.write(gpd.to_json())

    print(f'File saved to {geojson_path}')


if __name__ == '__main__':
    main()
