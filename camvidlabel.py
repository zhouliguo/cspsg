import glob
import cv2
import numpy as np

bgr_color = [[64,128,64],
[128,0,192],
[192,128,0],
[64,128,0],
[0,0,128],
[128,0,64],
[192,0,64],
[64,128,192],
[128,192,192],
[128,64,64],
[192,0,128],
[64,0,192],
[64,128,128],
[192,0,192],
[64,64,128],
[128,192,64],
[0,64,64],
[128,64,128],
[192,128,128],
[192,0,0],
[128,128,192],
[128,128,128],
[192,128,64],
[64,0,0],
[64,64,0],
[128,64,192],
[0,128,128],
[192,128,192],
[64,0,64],
[0,192,192],
[0,0,0],
[0,192,64]
]


label_path = glob.glob('D:/datasets/camvid/LabeledApproved_full/*')

for pt in label_path:
    img = cv2.imread(pt, cv2.IMREAD_UNCHANGED)
    img_gray = np.ones(img.shape[:2], np.uint8)*255
    for i, bgr in enumerate(bgr_color):
        index = np.where(img[:,:,0] == bgr[0]) and np.where(img[:,:,1] == bgr[1]) and np.where(img[:,:,2] == bgr[2])
        img_gray[index] = i
    #cv2.imshow('img_gray', img_gray)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    cv2.imwrite(pt[:-6]+'.png', img_gray)

