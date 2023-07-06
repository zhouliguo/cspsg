import glob
import os
import cv2
import numpy as np

def labelid():
    path = 'D:/datasets/cityscapes/gtFine/train_extra'
    image_path = glob.glob(os.path.join(path,'*','*leftImg8bit.png'), recursive=True)

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

if __name__=='__main__':

    image_path1 = glob.glob(os.path.join('D:/datasets/cityscapes/gtFine/train_extra','*','*labelTrainIds.png'), recursive=True)
    image_path2 = glob.glob(os.path.join('D:/datasets/cityscapes/gtCoarse/train_extra','*','*labelTrainIds.png'), recursive=True)
    image_path3 = glob.glob(os.path.join('D:/datasets/cityscapes/leftImg8bit/train_extra','*','*.png'), recursive=True)
    for path1, path2, path3 in zip(image_path1, image_path2, image_path3):
        image1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
        image3 = cv2.imread(path3, cv2.IMREAD_UNCHANGED)
        for i in range(19):
            index = np.where(image2==i)
            n = len(index[0])
            diff = np.sum(np.abs(image1[index]-image2[index]))
            print(diff)
            if diff>0:
                for j in range(len(index[0])):
                    if image1[index[0][j],index[1][j]] != image2[index[0][j],index[1][j]]:
                        image3[index[0][j],index[1][j]] = [0,0,255]
        cv2.imshow('1', image3)
        cv2.waitKey(33)
