import argparse
import time
from sys import platform
import os
import sys
sys.path.insert(0,'./model/yolov3')

from model.yolov3.models import *
from model.yolov3.utils.datasets import *
from model.yolov3.utils.utils import *
from app.split_image import split_image,merge_image,yolov3_loadImages,yolov3_loadVideo
import numpy as np
from easydict import EasyDict as edict

def filter_detection(det_results):
    """
    remove other object
    """
    target_objects=['person','bicycle','car','motorbike']
    det=det_results
    return det

def detect(
        cfg,
        data_cfg,
        weights,
        video_url='data/samples',  # input video url
        output='output',  # output folder
        save_result=False,
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        show_full_img=False,
):
    device = torch_utils.select_device()
    if save_result:
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (416, 416)  # onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return
    
    dataloader = yolov3_loadVideo(video_url,img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, resize_imgs, split_imgs,origin_img) in enumerate(dataloader):
        t = time.time()
        
        draw_imgs=[]
        det_results=[]
        for resize_img,split_img in zip(resize_imgs,split_imgs):
            # Get detections
            img = torch.from_numpy(resize_img).unsqueeze(0).to(device)
            pred, _ = model(img)
            det = non_max_suppression(pred, conf_thres, nms_thres)[0]
            
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
                
            det_results.append(det)
            if det is not None and len(det) > 0:
                # Rescale boxes from 416 to true image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], split_img.shape).round()
    
                # Print results to screen
                print('%gx%g ' % img.shape[2:], end='')  # print image size
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')
    
                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in det:    
                    # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, split_img, label=label, color=colors[int(cls)])
                
                if not show_full_img:
                    yield split_img
            if show_full_img:
                draw_imgs.append(split_img)
        if show_full_img:
            draw_img=merge_image(draw_imgs,img_size,origin_img.shape,det_results)
#        #result=non_max_suppression(det_results,conf_thres,nms_thres)[0]
#        print('Done. (%.3fs)' % (time.time() - t))
#        #draw_img,result
            yield draw_img

def yolov3_detect(video_url,show_full_img=False):
    opt=edict()
    opt.cfg='app/config/yolov3.cfg'
    opt.data_cfg='app/config/coco.data'
    opt.weights='app/config/yolov3.weights'
    opt.video_url=video_url
    opt.img_size=416
    with torch.no_grad():
        gen=detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            video_url=opt.video_url,
            img_size=opt.img_size,
            show_full_img=show_full_img,
        )
        
        idx=0
        for img in gen:
            if idx>30:
                idx=0
                yield img
            else:
                idx+=1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='app/config/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='app/config/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='app/config/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--video_url', type=str, default='rtsp://admin:juancheng1@221.1.215.254:554', help='full rtsp url')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save_result',default=False,action='store_true',help='load video or image')
    parser.add_argument('--output',default='',help='directory to save output result')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            video_url=opt.video_url,
            output=opt.output,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            save_result=opt.save_result
        )
