# deepCattleCounts

Python code to the CSRNet implementation used to detect and count cattle in the Amazon (Hodel et al.,in review)

This architecture and this code is adapted from 
+ [CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes,
  Li, Yuhong and Zhang, Xiaofan and Chen, Deming,Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018](https://arxiv.org/abs/1802.10062)
+ [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch.git)

## create a conda environment with gpu support

`conda env create -f torch_environment.yml`
`conda activate pytorch3.8`

## Estimate cattle distribution on VHR satellite images
Download the ensemble parameters [here] and put them in a folder `/parameters`. 
`python inference.py img.jpg img.kml parameters/`

Img.geojson file with geopoints for approximately every 400 px x 400 px 
containing the predicted cattle number and standard deviation from the estimate. 


## Testing 
In the jupyter notebook `Ensemble-test-set.ipynb` the ensemble of the trained CSRNet set is evaluated and an example 
image shown. The test set, as well as the full image dataset are available upon request.

## Training the model with new images
 
continue training the model with new data as following. From labels using 'labelImg', 
convert them into heatmaps as instructed in the jupyter notebook:
``

`python train.py train.json val.json 0 Adam_01`
