# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import sys
import random
sys.path.insert(0,'./model/yolov3')
from model.yolov3.models import Darknet,load_darknet_weights
from model.yolov3.utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box, bbox_iou
from model.yolov3.utils.parse_config import parse_data_cfg
from model.yolov3.utils.torch_utils import select_device

#from model.yolov3.utils.utils import bbox_iou,non_max_suppression
from app.split_image import split_image,yolov3_loadImages
from easydict import EasyDict as edict
import torch
import numpy as np

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
    if det is not None:
        det_idx=[]
        for c in det[:,-1]:
            if classes[int(c)] not in ['car','bicycle','motorbike','truck']:
                print('filter out',classes[int(c)])
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
        if batch_det is not None and len(batch_det)>0:
            batch_det[:, :4] = scale_coords(img.shape[2:], batch_det[:, :4], frame.shape).round()
            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in batch_det:
                # Add bbox to the image
                label = '%s %.2f' % (self.classes[int(cls)], conf)
                plot_one_box(xyxy, draw_origin_img, label=label, color=self.colors[int(cls)])
        return draw_origin_img,batch_det

    def process_slide(self,frame,conf_thres=0.5,nms_thres=0.5):
        frame=cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR);
        split_imgs=split_image(frame,self.img_size)
        resize_imgs=[self.preprocess(img) for img in split_imgs]
        batch_imgs=torch.stack([torch.from_numpy(img) for img in resize_imgs]).to(self.device)
        batch_pred,_=self.model(batch_imgs)
        #batch_det is a detection result list for img in batch_imgs
        batch_det=non_max_suppression(batch_pred, conf_thres, nms_thres)

        batch_det=[filter_label(det,self.classes,self.device) for det in batch_det]
        draw_origin_img=frame.copy()
        if batch_det is not None:
            merged_det=merge_bbox(batch_det,self.img_size,frame.shape[:2],conf_thres,nms_thres)
            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in merged_det:
                # Add bbox to the image
                label = '%s %.2f' % (self.classes[int(cls)], conf)
                plot_one_box(xyxy, draw_origin_img, label=label, color=self.colors[int(cls)])
        else:
            merged_det=None

        draw_origin_img=cv2.resize(draw_origin_img,(0, 0),fx=2.0,fy=2.0,interpolation=cv2.INTER_LINEAR)

        return draw_origin_img,merged_det

class MyVideoCapture():
    def __init__(self,rtsp_url):
        self.rtsp_url=rtsp_url
        self.cap=cv2.VideoCapture(rtsp_url)

    def read(self):
        flag,frame=self.cap.read()
        if flag:
            return True,frame
        else:
            print('restart video capture')
            self.cap.release()
            self.cap=cv2.VideoCapture(self.rtsp_url)
            assert self.cap.isOpened()
            return self.read()

    def release():
        self.cap.release()

def MyVideoWriter():
    def __init__(self):
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")
        #codec = cv2.VideoWriter_fourcc(*"h264")
        self.fps = 30
        self.writer=None
        self.video_path=None

    def get_writer(self,frame):
        """
        need release self.writer before write to new video
        """
        height,width=frame.shape[0:2]
        # todo load config from config file
        self.video_path=os.path.join('media',str(time.time())+".mp4")
        self.writer = cv2.VideoWriter(self.video_path,
                        self.codec, self.fps,
                        (width, height))
        self.frame=0

    def write(self,image):
        if self.writer is None:
            self.get_writer()

        if self.frame==30*60*10:
            self.writer.release()
            self.get_writer()

        self.writer.write(image)
        self.frame+=1

        return self.video_path



def rtsp2video(rtsp_url,video_path,save_minutes=10):
    reader=cv2.VideoCapture(rtsp_url)
    flag=reader.isOpened()
    if not flag:
        print('cannot open',rtsp_url)
        return -1

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    #codec = cv2.VideoWriter_fourcc(*"h264")
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(video_path, codec, fps, (width, height))

    opt=edict()
    opt.cfg='app/config/yolov3.cfg'
    opt.data_cfg='app/config/coco.data'
    opt.weights='app/config/yolov3.weights'
    opt.img_size=416
    detector=yolov3_slideWindows(opt)

    N=save_minutes*60*30
    for idx in range(N):
        flag,frame=reader.read()
        if flag:
            image,bbox=detector.process(frame)
            writer.write(image)
            print(idx,'save image to ',video_path)
        else:
            print(idx,'read frame failed!!!',rtsp_url)
            reader=MyVideoCapture(rtsp_url)
            frame=reader.read()

    reader.release()
    writer.release()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rtsp',
                        help='rtsp url address',
                        default='rtsp://admin:juancheng1@221.1.215.254:554',
                        )

    parser.add_argument('--video',
                        help='saved video name',
                        default='save.mp4')

    parser.add_argument('--save_minutes',
                        help='save video time',
                        type=int,
                        default=1)

    args=parser.parse_args()
    rtsp2video(args.rtsp,args.video,args.save_minutes)
