import numpy as np
import os
import cv2
import torch
from torch.nn import functional as F

from utils.dataloaders import LoadImagesAndLabels_sg_cam
from stream_metrics import StreamSegMetrics
import time

from labels import trainId2label

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

weights = 'runs/train/exp32/weights/152.pt'
val_path = 'C:\\datasets\\CamVid\\camvid_test.txt'
batch_size = 1
workers = 0
phase = 'test'  # val: original, test: original + fliplr

device = torch.device('cuda:0')
model = torch.load(weights, map_location='cpu').to(device)  # load checkpoint to CPU to avoid CUDA memory leak

val_dataset = LoadImagesAndLabels_sg_cam(val_path, phase=phase)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)

metrics = StreamSegMetrics(19)

model.eval()
metrics.reset()
with torch.no_grad():
    if phase == 'test':
        for i, (imgs, labels, imgs_f, _) in enumerate(val_loader):  # batch -------------------------------------------------------------
            imgs = torch.cat((imgs, imgs_f), dim=0)

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.numpy()

            outputs = model(imgs)[0]

            outputs = F.softmax(outputs, dim=3).detach().cpu().numpy()
            #outputs = outputs.detach().cpu().numpy()
            
            outputs[1] = np.fliplr(outputs[1])
            outputs = outputs[0] + outputs[1]
            #outputs = np.concatenate([outputs[0], outputs[1]], axis=2)
            preds = np.argmax(outputs, axis=2)
            #preds[preds>18] = preds[preds>18]-19

            '''
            probs, preds = torch.max(outputs, dim=3)
            probs, preds = probs.detach().cpu().numpy(), preds.detach().cpu().numpy()

            probs[1] = np.fliplr(probs[1])
            preds[1] = np.fliplr(preds[1])

            index = np.where(probs[0]<probs[1])
            preds[0][index] = preds[1][index]
            '''

            metrics.update(labels[0], preds)
            print(i)
    elif phase == 'val':
        sum_time = 0
        for i, (imgs, labels, paths) in enumerate(val_loader):  # batch -------------------------------------------------------------
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.numpy()
            start = time.time()
            outputs = model(imgs)[0]
            end = time.time()
            sum_time = sum_time+(end - start)
            preds = torch.argmax(outputs, dim=3).detach().cpu().numpy()[0]

            '''
            h, w = preds.shape
            preds1 = np.zeros((h, w))
            preds2 = np.zeros((h, w, 3))
            file_name = os.path.basename(paths[0])
            for j in range(19):
                preds1[preds==j] = trainId2label[j].id
                preds2[preds==j] = trainId2label[j].color
            cv2.imwrite('results/s/'+file_name, preds1)
            cv2.imwrite('results/s_color/'+file_name, preds2[:,:,::-1])
            '''

            metrics.update(labels[0], preds)
            print(i)
        print('Time:', sum_time)
        print('FPS:', len(val_dataset)/sum_time)
    elif phase == 'submit':
        for i, (imgs, labels, imgs_f, paths) in enumerate(val_loader):  # batch -------------------------------------------------------------
            imgs = torch.cat((imgs, imgs_f), dim=0)

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.numpy()

            outputs = model(imgs)[0]

            outputs = F.softmax(outputs, dim=3).detach().cpu().numpy()
            #outputs = outputs.detach().cpu().numpy()
            
            outputs[1] = np.fliplr(outputs[1])
            outputs = outputs[0] + outputs[1]
            #outputs = np.concatenate([outputs[0], outputs[1]], axis=2)
            preds = np.argmax(outputs, axis=2)

            h, w = preds.shape
            preds1 = np.zeros((h, w))
            preds2 = np.zeros((h, w, 3))
            file_name = os.path.basename(paths[0])
            for j in range(19):
                preds1[preds==j] = trainId2label[j].id
                preds2[preds==j] = trainId2label[j].color
            preds2[labels[0]==255] = [0,0,0]
            cv2.imwrite('results/l/'+file_name, preds1)
            cv2.imwrite('results/l_color/'+file_name, preds2[:,:,::-1])
            metrics.update(labels[0], preds)
            print(i)
    elif phase == 'speed':
        sum_time = 0
        input = torch.randn((1,3,1024,2048)).to(device, non_blocking=True)
        input.uniform_(0,1)
        for i in range(1000):  # batch -------------------------------------------------------------
            outputs = model(input)[0]
        start = time.time()
        for i in range(500):  # batch -------------------------------------------------------------
            outputs = model(input)[0]
        end = time.time()
        sum_time = sum_time+(end - start)
        print('Time:', sum_time)
        print('FPS:', len(val_dataset)/sum_time)
    else:
        print(1)

    score = metrics.get_results()
    accuracy = score["Overall Acc"]
    miou = score["Mean IoU"]
            
    print('Accuracy:', accuracy, 'Miou:', miou)
    torch.cuda.empty_cache()
    #Accuracy: 0.9649460363756696 Miou: 0.8116732825134334
    #Accuracy: 0.9666793166096579 Miou: 0.8183937872003023
    #Accuracy: 0.966843847471226 Miou: 0.8181737471207283

    #L_f: Accuracy: 0.9693415456790155 Miou: 0.8291020889999201
    #L:   Accuracy: 0.9687497123996338 Miou: 0.8246733637132168

    #M_f: Accuracy: 0.9691801518861988 Miou: 0.8225829821685342
    #M:   Accuracy: 0.968605896397818 Miou: 0.8197655137174934

    #S_f: Accuracy: 0.9661207440713181 Miou: 0.8044007063347617 
    #S:   Accuracy: 0.9654496278832546 Miou: 0.8012256461027449