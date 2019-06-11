# -*- coding: utf-8 -*-

import os
import cv2
import argparse

def process(frame,index):
    img_name='sy'+'%06d'%index+'.jpg'
    cv2.imwrite(img_name,frame)
    print('save image to',img_name)

def rtsp2video(rtsp_url,video_path):
    reader=cv2.VideoCapture(rtsp_url)
    flag=reader.isOpened()
    if not flag:
        print('cannot open',rtsp_url)
        return -1
    
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    fps = reader.get(cv2.CAP_PROP_FPS)
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(video_path, codec, fps, (width, height))
        
    for idx in range(100):
        flag,frame=reader.read()
        if flag:
            writer.write(frame)
            print('write frame from',rtsp_url,'to',video_path)
        else:
            print(idx,'read frame failed!!!',rtsp_url)
        
    reader.release()
    writer.release()
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rtsp',
                        help='rtsp url address',
                        )
    
    parser.add_argument('--video',
                        help='saved video name',
                        default='save.mp4')
    
    args=parser.parse_args()
    rtsp2video(args.rtsp,args.video)