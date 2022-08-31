# A Fast and Accurate Semantic Segmentation Method

## Download Datasets

### Cityscapes [[1]](#1)

#### Website
https://www.cityscapes-dataset.com/

#### Download by Wget [[2]](#2) (if you need)

```
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=yourusername&password=yourpassword&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
```
In the first line, put your username and password. This will login with your credentials and keep the associated cookies.

In the second line, you need to provide the packageID paramater and it downloads the file.

packageIDs map like this in the website:
```
1 -> gtFine_trainvaltest.zip (241MB)
2 -> gtCoarse.zip (1.3GB)
3 -> leftImg8bit_trainvaltest.zip (11GB)
4 -> leftImg8bit_trainextra.zip (44GB)
8 -> camera_trainvaltest.zip (2MB)
9 -> camera_trainextra.zip (8MB)
10 -> vehicle_trainvaltest.zip (2MB)
11 -> vehicle_trainextra.zip (7MB)
12 -> leftImg8bit_demoVideo.zip (6.6GB)
28 -> gtBbox_cityPersons_trainval.zip (2.2MB)
```

## Prepare Training Label

1. Download cityscapesScripts https://github.com/mcordts/cityscapesScripts
2. Modify 'cityscapesPath' in 'cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py'
3. Run python createTrainIdLabelImgs.py

## Train
1. Modify 'path' in 'cspsg\data\cityscapes.yaml'
2. Modify --data, --batch-size, --device ... in parse_opt of 'cspsg\train_sg.py'
3. python train_sg.py (single GPU), python -m torch.distributed.run --nproc_per_node N train.py --sync-bn (N GPUs)

## References
<a id="1">[1]</a> M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 

<a id="2">[2]</a> https://towardsdatascience.com/download-city-scapes-dataset-with-script-3061f87b20d7