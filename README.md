# deepCattleCounts
 Python code to the CSRNet implementation used to detect and count cattle in the Amazon (Hodel et al.,in review)
 
 
## 

Python: 3.9

PyTorch: 1.12.1

CUDA: 11.7

This architecture and this code is adapted from 
+ [CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes,
  Li, Yuhong and Zhang, Xiaofan and Chen, Deming,Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018](https://arxiv.org/abs/1802.10062)
+ [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch.git)

# create a conda environment with gpu support
conda create --name pytorch-gpu --file requirements.txt

## Training and validating CSRNet to count cattle on image chips
 
continue training the model with new data as following. put path to new labeled images
look at the jupyter notebook 
``
to figure out how to create a new training set and how to prepare the 
json files with the path to the new labeled images. then run

 `python train.py train.json val.json 0 Adam_01`

 
## Testing and Inference on new data

The full image dataset as well as the ground truth test set is available upon request

You can  download the emsemble parametres [here (not public yet)]().


Then run:

`python inference.py inference_list.json parameters/`
 
## Output

The output of the inference on all images can also be directly downloaded [here (not public yet)]()
