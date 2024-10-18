# deepCattleCount

With deep learning-based cattle counts on satellite imagery we offer evidence regarding land use and policy impact in the Brazilian Amazon.
This repository includes the python code to the CSRNet implementation (Hodel et al.,in review)

![](./imgs/figure1.png)

This architecture and this code is adapted from 
+ [CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes,
  Li, Yuhong and Zhang, Xiaofan and Chen, Deming,Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018](https://arxiv.org/abs/1802.10062)
+ [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch.git)

## Create a conda environment

`conda env create -f environment.yml`

`conda activate deepcattlecount`

## Downloads

Download [pre-trained weights](https://zenodo.org/records/13385687) for inference on new images.

## Estimate cattle distribution on VHR satellite images

This model is designed to perform inference on very high-resolution satellite images with a spatial resolution of 28 cm per pixel. 
It utilizes both a JPEG file containing RGB data and a KML file that provides the geospatial context for the image.

`python inference.py parameters/ pathto/img.jpg pathto/img.kml`

The output of the model is an Img.geojson file, which includes geospatial points corresponding to approximately every 400 x 400 pixel segment of the input image. 
This geospatial point contains the predicted number of cattle and the ensemble-generated standard deviation of the estimates.

## Training 

the files train.json and test.json contain the paths to the individual train and test images. 
Trained parameters will be saved in the parameters folder.

`python train.py --train_json train.json --test_json test.json 0 parmeters/parameters1`


## Testing 

In the jupyter notebook [Ensemble-test-set.ipynb](https://github.com/leoniehodel/deepCattleCount/blob/master/Ensemble-test-set.ipynb) the ensemble of the trained CSRNet set is evaluated and an example 
image shown. 

The test set, as well as the full image dataset are available upon request.

## Dataset analysis

The code for the statistical analysis of the paper, as well as the cattle maps can be found 
under [/leoniehodel/deepCattleCounts_analysis](https://github.com/leoniehodel/deepCattleCount_analysis/)
