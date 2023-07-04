import glob
import os
import cv2
import numpy as np

path = '/mnt/ssd/cityscapes/gtFine/train_extra'
image_path = glob.glob(os.path.join(path,'*','*.png'), recursive=True)

for pt in image_path:
    print(pt[:-15])
    image = cv2.imread(pt, cv2.IMREAD_UNCHANGED)
    h, w = image.shape
    image_new = np.zeros((h, w), np.uint8)+255

    image_new[image==7] = 0
    image_new[image==8] = 1
    image_new[image==11] = 2
    image_new[image==12] = 3
    image_new[image==13] = 4
    image_new[image==17] = 5
    image_new[image==19] = 6
    image_new[image==20] = 7
    image_new[image==21] = 8
    image_new[image==22] = 9
    image_new[image==23] = 10
    image_new[image==24] = 11
    image_new[image==25] = 12
    image_new[image==26] = 13
    image_new[image==27] = 14
    image_new[image==28] = 15
    image_new[image==31] = 16
    image_new[image==32] = 17
    image_new[image==33] = 18
    pt = pt[:-15] + 'gtCoarse_labelTrainIds.png'
    cv2.imwrite(pt, image_new)
