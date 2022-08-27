# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        #BCEcls = nn.CrossEntropyLoss()
        #BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.gr, self.hyp, self.autobalance = BCEcls, 1.0, h, autobalance
        # self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        # self.anchors = m.anchors
        self.device = device
        #self.ifratio = torch.tensor([8,16,32],device=device) #image/feature

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        #lbox = torch.zeros(1, device=self.device)  # box loss
        #lobj = torch.zeros(1, device=self.device)  # object loss
        indices = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
            # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
            pcls = pi[indices]
            tcls = targets[indices]

            # Regression
            #pxy = pxy.sigmoid() * 2 - 0.5
            #pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
            #pwh = torch.pow(anchors[i], pwh.sigmoid() + 1)/self.ifratio[i]
            #pbox = torch.cat((pxy, pwh), 1)  # predicted box
            #iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
            #lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    #iou = iou.detach().clamp(0).type(tobj.dtype)
                    #if self.sort_obj_iou:
                    #    j = iou.argsort()
                    #    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                    #if self.gr < 1:
                    #    iou = (1.0 - self.gr) + self.gr * iou
                    #tobj[b, a, gj, gi] = iou  # iou ratio

                    # Classification
                    #if self.nc > 1:  # cls loss (only if multiple classes)
            #t = torch.full_like(pcls, self.cn, device=self.device)  # targets
            #t[:, tcls] = self.cp
            lcls += self.BCEcls(pcls, tcls)  # BCE

                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            #obji = self.BCEobj(pi[..., 4], tobj)
            #lobj += obji * self.balance[i]  # obj loss
            #if self.autobalance:
            #    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        #if self.autobalance:
        #    self.balance = [x / self.balance[self.ssi] for x in self.balance]
        #lbox *= self.hyp['box']
        #lobj *= self.hyp['obj']
        #lcls *= self.hyp['cls']
        bs = pi.shape[0]  # batch size

        #return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        return lcls * bs, lcls.detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets[:,:,:,0]
        indices = torch.where(targets!=-1)
        return indices

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            
            gi, gj = [], []
            gx, gy, gw, gh = [], [], [], []
            a, b, c = [], [], []

            if nt:
                #r = t[:, :, 4:6]*anchors[0]
                r = t[:, :, 4:6]*self.ifratio[i]
                r = torch.log(r)/torch.log(anchors[0])

                j = torch.max(r[:,:,0], r[:,:,1]) <= 2
                k = torch.min(r[:,:,0], r[:,:,1]) >= 1
                #k = torch.min(t[:,:,4], t[:,:,5]) >= 1
                j = torch.logical_and(j, k)
                t = t[j]  # filter

                '''
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                '''
                
                for ti in t:
                    
                    px1 = ti[2].int()
                    py1 = ti[3].int()

                    if torch.round(ti[2]) == px1:
                        px2 = px1-1
                    else:
                        px2 = px1+1
                    py2 = py1

                    if torch.round(ti[3]) == py1:
                        py3 = py1-1
                    else:
                        py3 = py1+1
                    px3 = px1

                    #px4 = px2
                    #py4 = py3
                    
                    px = []#torch.stack((px1, px2, px3)).clamp_(0, gain[2] - 1)
                    py = []#torch.stack((py1, py2, py3)).clamp_(0, gain[3] - 1)

                    
                    #gl = torch.round(ti[2]-ti[4]/2)
                    #gr = torch.round(ti[2]+ti[4]/2)-1
                    #gu = torch.round(ti[3]-ti[5]/2)
                    #gd = torch.round(ti[3]+ti[5]/2)-1

                    gl = (ti[2]-ti[4]/2).int()
                    gr = (ti[2]+ti[4]/2).int()
                    gu = (ti[3]-ti[5]/2).int()
                    gd = (ti[3]+ti[5]/2).int()

                    if gr < gl:
                        gl = ti[2].int()
                        gr = ti[2].int()
                    if gd < gu:
                        gu = ti[3].int()
                        gd = ti[3].int()

                    if px1>=gl and px1<=gr and py1>=gu and py1<=gd:
                        px.append(px1)
                        py.append(py1)
                    if px2>=gl and px2<=gr and py2>=gu and py2<=gd:
                        px.append(px2)
                        py.append(py2)
                    if px3>=gl and px3<=gr and py3>=gu and py3<=gd:
                        px.append(px3)
                        py.append(py3)

                    #gii = torch.arange(gl, gr+1).clamp_(0, gain[2] - 1).cuda()
                    #gjj = torch.arange(gu, gd+1).clamp_(0, gain[3] - 1).cuda()
                    
                    gi.append(torch.stack(px))
                    gj.append(torch.stack(py))

                    px_len = len(px)
                    gx.append(ti[2].repeat(px_len) - torch.stack(px))
                    gy.append(ti[3].repeat(px_len) - torch.stack(py))
                    gw.append(ti[4].repeat(px_len))
                    gh.append(ti[5].repeat(px_len))
                    
                    a.append(ti[6].repeat(px_len))
                    b.append(ti[0].repeat(px_len))
                    c.append(ti[1].repeat(px_len))

                if len(gi)>0 and len(gj)>0:
                    gi = torch.cat(gi, -1).clamp_(0, gain[2] - 1).long()
                    gj = torch.cat(gj, -1).clamp_(0, gain[3] - 1).long()
                    gx = torch.cat(gx, -1)
                    gy = torch.cat(gy, -1)
                    gw = torch.cat(gw, -1)
                    gh = torch.cat(gh, -1)
                    a = torch.cat(a, -1).long()
                    b = torch.cat(b, -1).long()
                    c = torch.cat(c, -1).long()
            #else:
            #    return tcls, tbox, indices, anch
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            #tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            if len(gi)>0 and len(gj)>0:
                tbox.append(torch.stack((gx, gy, gw, gh)).T)  # box
            else:
                tbox.append((gx, gy, gw, gh))
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch