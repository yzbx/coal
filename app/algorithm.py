# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import sys
import random
import time
sys.path.insert(0,'./model/yolov3')
from model.yolov3.models import Darknet,load_darknet_weights
from model.yolov3.utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box, bbox_iou
from model.yolov3.utils.parse_config import parse_data_cfg
from model.yolov3.utils.torch_utils import select_device
from torchvision.models import vgg11

#from model.yolov3.utils.utils import bbox_iou,non_max_suppression
from app.split_image import split_image,yolov3_loadImages
from easydict import EasyDict as edict
import torch
import numpy as np
import requests
import warnings
import logging

def my_nms(pred,nms_thres=0.5):
    """
    do the second time nms for slide windows results
    input format: tensor [x1,y1,x2,y2]
    """
    # Get detections sorted by decreasing confidence scores
    pred = pred[(-pred[:, 4]).argsort()]

    det_max = []
    nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
    for c in pred[:, -1].unique():
        dc = pred[pred[:, -1] == c]  # select class c
        n = len(dc)
        if n == 1:
            det_max.append(dc)  # No NMS required if only 1 prediction
            continue
        elif n > 100:
            dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

        # Non-maximum suppression
        if nms_style == 'OR':  # default
            while dc.shape[0]:
                det_max.append(dc[:1])  # save highest conf detection
                if len(dc) == 1:  # Stop if we're at the last detection
                    break
                iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                dc = dc[1:][iou < nms_thres]  # remove ious > threshold

        elif nms_style == 'AND':  # requires overlap, single boxes erased
            while len(dc) > 1:
                iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                if iou.max() > 0.5:
                    det_max.append(dc[:1])
                dc = dc[1:][iou < nms_thres]  # remove ious > threshold

        elif nms_style == 'MERGE':  # weighted mixture box
            while len(dc):
                if len(dc) == 1:
                    det_max.append(dc)
                    break
                i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                weights = dc[i, 4:5]
                dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                det_max.append(dc[:1])
                dc = dc[i == 0]

        elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
            sigma = 0.5  # soft-nms sigma parameter
            while len(dc):
                if len(dc) == 1:
                    det_max.append(dc)
                    break
                det_max.append(dc[:1])
                iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                dc = dc[1:]
                dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences

    if len(det_max):
        det_max = torch.cat(det_max)  # concatenate
        output = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output

def merge_bbox(bboxes,target_size,origin_size,conf_thres=0.5,nms_thres=0.5):
    """
    use slide window technology
    bboxes: small image's detection results
    target_size: small image normed size
    origin_size: full image size
    """
    if isinstance(target_size,int):
        target_size=(target_size,target_size)

    h,w=origin_size[0:2]
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1

    merged_bbox=[]
    for i in range(h_num):
        for j in range(w_num):
            if i==h_num-1:
                h_end=h
            else:
                h_end=i*th+2*th

            if j==w_num-1:
                w_end=w
            else:
                w_end=j*tw+2*tw

            idx=i*w_num+j
            # image size in slide window technology >= target_size
            shape=(h_end-i*th,w_end-j*tw)
            offset=(th*i,tw*j)
            if bboxes[idx] is not None:
                det=bboxes[idx]
                if target_size!=shape:
                    # Rescale boxes from target size to slide window size
                    y_h_scale=shape[0]/target_size[0]
                    x_w_scale=shape[1]/target_size[1]
                    det[:,[0,2]] =(det[:,[0,2]]*x_w_scale).round()
                    det[:,[1,3]]=(det[:,[1,3]]*y_h_scale).round()
                    det[:,0:4]=det[:,0:4].clamp(min=0)
                det[:,:4]+=torch.tensor([offset[1],offset[0],offset[1],offset[0]]).to(det)

                merged_bbox.append(det)

    if len(merged_bbox)==0:
        return []

    merged_bbox=torch.cat(merged_bbox,dim=0)
    nms_merged_bbox = my_nms(merged_bbox, nms_thres)

    return nms_merged_bbox

def filter_label(det,classes,device):
    if 'bicycle' in classes:
        filter_classes=['car','bicycle','motorbike','truck','person']
    else:
        return det

    if det is not None:
        det_idx=[]
        for c in det[:,-1]:
            if classes[int(c)] not in filter_classes:
                det_idx.append(0)
            else:
                det_idx.append(1)
        if np.any(det_idx):
            det=det[torch.from_numpy(np.array(det_idx)).to(device).eq(1),:]
        else:
            det=None

    return det

class yolov3_slideWindows(yolov3_loadImages):
    def __init__(self,opt):
        super().__init__(None,img_size=opt.img_size,preprocess=True)
        self.opt=opt
        self.classes=load_classes(parse_data_cfg(opt.data_cfg)['names'])
        self.colors=[[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        self.device = select_device()
        self.model=self.load_model()

    def load_model(self):
        model = Darknet(self.opt.cfg,self.img_size)
        # Load weights
        if self.opt.weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(self.opt.weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, self.opt.weights)
        # Fuse Conv2d + BatchNorm2d layers
        model.fuse()

        # Eval mode
        model.to(self.device).eval()

        return model

    def process(self,frame,conf_thres=0.5,nms_thres=0.5):
        img=torch.from_numpy(self.preprocess(frame)).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        batch_det=filter_label(det,self.classes,self.device)
        draw_origin_img=frame.copy()
        det_dicts=[]
        if batch_det is not None and len(batch_det)>0:
            batch_det[:, :4] = scale_coords(img.shape[2:], batch_det[:, :4], frame.shape).round()
            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in batch_det:
                # Add bbox to the image
                label = '%s %.2f' % (self.classes[int(cls)], conf)
                plot_one_box(xyxy, draw_origin_img, label=label, color=self.colors[int(cls)])
                det_dicts.append({'bbox':[float(x.detach()) for x in xyxy],'conf':float(conf.detach()),'label':self.classes[int(cls)]})
        return draw_origin_img,det_dicts

    def process_slide(self,frame,conf_thres=0.5,nms_thres=0.5):
        if min(frame.shape[0:2]) <= max(self.img_size):
            warnings.warn("for small image, forbid slide window technology")
            return self.process(frame,conf_thres,nms_thres)

        if min(frame.shape[0:2])<=2*max(self.img_size):
            resize_input=False
        else:
            resize_input=True

        if resize_input:
            frame=cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR);
        split_imgs=split_image(frame,self.img_size)
        # if len(split_imgs)==0:
        #     return cv2.resize(frame,(0, 0),fx=2.0,fy=2.0,interpolation=cv2.INTER_LINEAR),None
        resize_imgs=[self.preprocess(img) for img in split_imgs]
        batch_imgs=torch.stack([torch.from_numpy(img) for img in resize_imgs]).to(self.device)
        batch_pred,_=self.model(batch_imgs)
        #batch_det is a detection result list for img in batch_imgs
        batch_det=non_max_suppression(batch_pred, conf_thres, nms_thres)

        batch_det=[filter_label(det,self.classes,self.device) for det in batch_det]
        draw_origin_img=frame.copy()

        det_dicts=[]
        if batch_det is not None:
            merged_det=merge_bbox(batch_det,self.img_size,frame.shape[:2],conf_thres,nms_thres)

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in merged_det:
                # Add bbox to the image
                label = '%s %.2f' % (self.classes[int(cls)], conf)
                plot_one_box(xyxy, draw_origin_img, label=label, color=self.colors[int(cls)])
                det_dicts.append({'bbox':[float(x.detach()) for x in xyxy],'conf':float(conf.detach()),'label':self.classes[int(cls)]})
        else:
            merged_det=None

        if resize_input:
            draw_origin_img=cv2.resize(draw_origin_img,(0, 0),fx=2.0,fy=2.0,interpolation=cv2.INTER_LINEAR)
        return draw_origin_img,det_dicts

def simple_preprocess(image,img_size):
    # Padded resize
    img=cv2.resize(image,tuple(img_size),interpolation=cv2.INTER_LINEAR)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img

class vgg_fire():
    def __init__(self,opt):
        self.opt=opt
        self.model=self.load_model()

    def load_model(self):
        model=vgg11(pretrained=False,num_classes=2)
        model_path=self.opt.weights
        model.load_state_dict(torch.load(model_path).state_dict())
        model.eval()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        return model

    def process(self,frame,conf_thres=0.5,nms_thres=None):
        if nms_thres:
            warnings.warn('nms_thres not work for classification model vgg_fire')


        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_size=(224,224)
        img=simple_preprocess(frame,img_size)
        inputs=torch.unsqueeze(torch.from_numpy(img),dim=0).to(device)
        outputs=self.model.forward(inputs)
        result=torch.softmax(outputs,dim=1).data.cpu().numpy()
        result=np.squeeze(result)

        names=['normal','fire']
        text=names[np.argmax(result)]
        conf=float(max(result))
        if text==names[1]:
            color=(0,0,255)
        else:
            color=(255,0,0)

        # convert image to [height width channel] format
        fontScale=max(1,frame.shape[1]//448)
        thickness=max(1,frame.shape[1]//112)
        frame=cv2.putText(frame, text+' %0.2f'%(conf) , (50,50), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness)

        return frame,[{'bbox':[0,0,0,0],'conf':conf,'label':text}]

    def process_slide(self,**args):
        return self.process(args)


