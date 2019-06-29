# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import warnings

def split_image(image,target_size,draw_split=False):
    """
    apply slide window technology on image
    target_size=[th,tw]
    return images with size >= target_size
    """
    if isinstance(target_size,int):
        target_size=(target_size,target_size)

    h,w,c=image.shape
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1

    imgs=[]
    draw_img=image
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


            if draw_split:
                pt1=(j*tw,i*th)
                pt2=(w_end,h_end)
                draw_img=cv2.rectangle(draw_img,pt1,pt2,color=(255,0,0),thickness=5)
            else:
                img=image[i*th:h_end,j*tw:w_end]
                imgs.append(img)

    if draw_split:
        return draw_img
    else:
        if len(imgs)==0:
            warnings.warn(str(image.shape)+' '+str(target_size))
        return imgs

def merge_image(imgs,target_size,origin_size):
    if isinstance(target_size,int):
        target_size=(target_size,target_size)

    h,w,c=origin_size
    th=target_size[0]//2
    tw=target_size[1]//2
    h_num=int(np.floor(h/th))-1
    w_num=int(np.floor(w/tw))-1

    image=np.zeros(origin_size,np.uint8)
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

            split_img=imgs[i*w_num+j]
            shape=(h_end-i*th,w_end-j*tw)
            if split_img.shape[0:2]!=shape:
                new_img=cv2.resize(split_img,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
                image[i*th:h_end,j*tw:w_end]=new_img
            else:
                image[i*th:h_end,j*tw:w_end]=split_img

    return image

def preprocess(pre_img,img_size,do_preprocess):
    """
    preprocess image for yolov3 network
    """
    # Padded resize
    img=cv2.resize(pre_img,tuple(img_size),interpolation=cv2.INTER_LINEAR)

    # Normalize RGB
    if do_preprocess:
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img

class yolov3_loadImages:
    def __init__(self,path,img_size=[416,416],preprocess=True):
        if isinstance(img_size,int):
            img_size=(img_size,img_size)

        self.img_size=img_size
        self.do_preprocess=preprocess
        if path is not None:
            files=glob.glob(os.path.join(path,'**','*'),recursive=True)
            suffix=('jpg','png','jpeg','bmp')
            self.img_files=[f for f in files if f.lower().endswith(suffix)]
            self.count=0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.img_files)

    def __next__(self):
        if self.count>=len(self.img_files):
            raise StopIteration

        path=self.img_files[self.count]
        origin_img=cv2.imread(path)
        self.count+=1
        split_imgs=split_image(origin_img,self.img_size)

        resize_imgs=[self.preprocess(img) for img in split_imgs]
        return path,resize_imgs,split_imgs,origin_img

    def preprocess(self,pre_img):
        return preprocess(pre_img,self.img_size,self.do_preprocess)

class yolov3_loadVideo(yolov3_loadImages):
    def __init__(self,video_url,img_size=[416,416],preprocess=True):
        super().__init__(None,img_size,preprocess)
        self.video_file=video_url
        self.cap=cv2.VideoCapture(self.video_file)
        self.frame=0

    def __len__(self):
        return 1

    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration

        success, origin_img = self.cap.read()
        if not success:
            raise StopIteration

        split_imgs=split_image(origin_img,self.img_size)

        resize_imgs=[self.preprocess(img) for img in split_imgs]
        return self.video_file,resize_imgs,split_imgs,origin_img

class My_VideoWriter():
    def __init__(self):
        self.current_video_path=None
        self.current_video_writer=None

    def new_writer(self,save_path,image):
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        fps=5
        height,width,_=image.shape
        dirname=os.path.dirname(save_path)
        os.makedirs(dirname,exist_ok=True)
        self.current_video_writer = cv2.VideoWriter(save_path, codec, fps, (width, height))
        self.current_video_path=save_path

    def write(self,save_path,image):
        if self.current_video_writer is None:
            self.new_writer(save_path,image)
        elif self.current_video_path != save_path:
            self.current_video_writer.release()
            self.new_writer(save_path,image)

        self.current_video_writer.write(image)

    def release(self):
        if self.current_video_writer is not None:
            self.current_video_writer.release()

if __name__ == '__main__':
    files=glob.glob(os.path.join('dataset/demo','**','*'),recursive=True)
    suffix=('jpg','png','jpeg','bmp')
    img_files=[f for f in files if f.lower().endswith(suffix)]

    target_size=(224,224)
    for img_f in img_files:
        image=cv2.imread(img_f)
        draw_img=split_image(image,target_size,draw_split=True)
        plt.imshow(draw_img)
        plt.show()

        break
