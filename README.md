# deepCattleCounts [in progress]

Python code to the CSRNet implementation used to detect and count cattle in the Amazon (Hodel et al.,in review)

This architecture and this code is adapted from 
+ [CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes,
  Li, Yuhong and Zhang, Xiaofan and Chen, Deming,Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018](https://arxiv.org/abs/1802.10062)
+ [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch.git)

## Create a conda environment with gpu support

`conda env create -f torch_environment.yml`

`conda activate pytorch-gpu`

## Estimate cattle distribution on VHR satellite images
Download the ensemble parameters [here] and put them in a folder `/parameters`. 
The satellite has to be a VHR satellite image 
`python inference.py parameters/ pathto/img.jpg pathto/img.kml`

Img.geojson file with geopoints for approximately every 400 px x 400 px 
containing the predicted cattle number and ensemble-generated standard deviation from the estimate. 

## Testing 
In the jupyter notebook `Ensemble-test-set.ipynb` the ensemble of the trained CSRNet set is evaluated and an example 
image shown. The test set, as well as the full image dataset are available upon request.
