import os
import json
from easydict import EasyDict as edict
import requests
from multiprocessing import Process, Value
class QD_Basic():
    def __init__(self,cfg):
        if isinstance(cfg,(string)):
            self.cfg=edict(json.loads(cfg))
        elif isinstance(cfg,(dict)):
            self.cfg=edict(cfg)
        elif isinstance(cfg,(edict)):
            self.cfg=cfg
        else:
            raise Exception('unknown cfg type')
    
class QD_Reader():
    def __init__(self,video_url):
        self.video_url=video_url
        self.max_retry_times=10
        self.current_retry_times=0
        self.cap=None
        
    def read(self):
        if self.cap is None:
            self.cap=cv2.VideoCapture(self.video_url)
        
        flag,frame=self.cap.read()
        if flag:
            return True,frame
        else:
            print('restart video capture')
            self.cap.release()
            
            time.sleep(0.5)
            self.cap=cv2.VideoCapture(self.video_url)
            self.retry_times+=1
            if self.retry_times>self.max_retry_times:
                raise StopIteration('retry times > {}'.format(self.max_retry_times))

            if not self.cap.isOpened():
                raise StopIteration('cannot open {}'.format(self.video_url))
                return False,None
            else:
                return self.read()
            
    def update_video_url(self,video_url):
        if self.cap is not None:
            self.cap.release()
        self.video_url=video_url
        self.cap=cv2.VideoCapture(self.video_url)
        
class QD_Writer():
    """
    cannot save all frame in memory
    5min, 30fps 1024x2048 RGB image need memory:
    5*60*30*1024x2048x3 > 30G
    """
    def __init__(self,filenames,save_frame_number):
        self.save_video_name=str(time.time())+'.mp4'
        self.save_frame_number=save_frame_number
        
        valid_num=self.save_frame_number//2
        self.image_names=filenames[-valid_num:]
    
    def write_sync(self,filename):
        if not self.can_upload():
            self.image_names.append(filename)
        else:
            raise Exception('no need to write frame when can upload')
            
    def can_upload(self):
        if len(self.image_names)<self.save_frame_number:
            return True
        else:
            return False
        
    def upload(self):
        pass
        
class QD_Detector(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        
    def process(self,frame):
        return frame,None


class QD_Alerter():
    def __init__(self):
        self.filenames=[]
        self.writers=[]
        self.save_frame_number=5*60*30
        self.max_filesize=self.save_frame_number*2
    
    def bbox2relu(self,bbox):
        return None
    
    def process(self,image,bbox):
        filename=str(time.time())+'.jpg'
        cv2.imwrite(filename,image)
        self.filenames.append(filename)
        
        if len(self.filenames)>self.max_filesize:
            self.filenames=self.filenames[-self.save_frame_number:]
        
        for writer in self.writers:
            if writer.can_uploaded():
                self.writer.upload()
                self.writers=self.writers[1:]
            else:
                self.writer.add(filename)
        
        if self.bbox2relu(bbox):
            #todo
            self.writers.append(QD_Writer(self.filenames,self.save_frame_number))
        
class QD_Process(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.reader=QD_Reader(self.cfg.video_url)
        self.detector=QD_Detector(self.cfg)
        self.alerter=QD_Alerter()
        
    def process(self):
        while True:
            flag,frame=self.reader.read()
            if flag:
                image,bbox=self.detector.process(frame)
                self.alerter.process(image,bbox)
            else:
                raise StopIteration('read frame failed!!!')
                break
    
    def demo(self):
        while True:
            flag,frame=self.reader.read()
            if flag:
                image,bbox=self.detector.process(frame)
                yield image
            else:
                raise StopIteration('read frame failed!!!')
                break
    
class QD_Upload(QB_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        with open('config.json','r') as f:
            config=json.load(f)
        self.upload_url=config.upload_url
        
    def upload(self,filename):
        """
        return example: 
        {"fileIp":"10.50.200.107:8888",
        "fileUrl":"group1/M00/00/00/CjLIa10V-92AUONHAAAACV_xtOE8838105",
        "success":true}
        
        {"success":false}
        """
        with open(filename,'rb') as f:
            files = {'upload': f}
            r = requests.post(self.video_url, files=files)
            r.close()
            result=json.loads(r.content)
            
            if result['success']:
                return result['fileUrl']
            else:
                warnings.warn('upload {} failed'.format(filename))
                return ''
            
        return ''
