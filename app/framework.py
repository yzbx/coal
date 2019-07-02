import os
import cv2
import json
import time
from easydict import EasyDict as edict
import requests
from multiprocessing import Process, Manager
import sys
if '.' not in sys.path:
    sys.path.insert(0,'.')
sys.path.insert(0,'./model/yolov3')
from videowrite import yolov3_slideWindows


class QD_Basic():
    def __init__(self,cfg):
        if isinstance(cfg,(str)):
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
    def __init__(self,rule,filenames,save_frame_number):
        self.rule=rule
        self.save_video_name=str(time.time())+'.mp4'
        self.save_frame_number=save_frame_number
        
        valid_num=self.save_frame_number//2
        self.image_names=filenames[-valid_num:]
        self.sub_process=None
        self.manage_dict=None
    
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
    
    def upload_in_subprocess(self):
        with Manager() as manager:
            self.manage_dict=manager.dict()
            self.sub_process=Process(target=save_and_upload,args=(self.image_names,self.save_video_name,self.manage_dict))
            self.sub_process.start()
            
        
class QD_Detector(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        opt=edict()
        opt.cfg='app/config/yolov3.cfg'
        opt.data_cfg='app/config/coco.data'
        opt.weights='app/config/yolov3.weights'
        opt.img_size=416
        self.detector=yolov3_slideWindows(opt)
        
    def process(self,frame):
        image,bbox=self.detector.process_slide(frame)
        return image,bbox


class QD_Alerter():
    """
    save image to disk and record the filename
    """
    def __init__(self,save_frame_number):
        self.filenames=[]
        self.writers=[]
        self.save_frame_number=save_frame_number
        self.max_filesize=self.save_frame_number*2
    
    def bbox2rule(self,bbox):
        """
        bbox: [{'bbox':list(xyxy),'conf':conf,'label':self.classes[int(cls)]}]
        convert bbox to object count dict
        """
        rule={}
        for b in bbox:
            label=b['label']
            if label not in rule.keys():
                rule[label]=1
            else:
                rule[label]+=1
                
        if rule:
            return json.dumps(rule)
        else:
            return None
    
    def process(self,image,bbox):
        filename=str(time.time())+'.jpg'
        cv2.imwrite(filename,image)
        self.filenames.append(filename)
        
        if len(self.filenames)>self.max_filesize:
            self.filenames=self.filenames[-self.save_frame_number:]
        
        for idx,writer in enumerate(self.writers):
            if writer.can_upload():
                if writer.sub_process is None:
                    writer.upload_in_subprocess()
                elif writer.sub_process.is_alive():
                    print('{} is running'.format(writer.sub_process.pid))
                else:
                    writer.sub_process.join()
                    self.writers[idx]=None
                    print('fileUrl is {fileUrl}'.format(fileUrl=writer.manage_dict['fileUrl']))
            else:
                writer.write_sync(filename)
        
        self.writers=[w for w in self.writers if w is not None]
        rule=self.bbox2rule(bbox)
        if rule:
            writer=QD_Writer(rule,self.filenames,self.save_frame_number)
            self.writers.append(writer)
        
class QD_Process(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.reader=QD_Reader(self.cfg.video_url)
        self.detector=QD_Detector(self.cfg)
        self.alerter=QD_Alerter(self.cfg.save_frame_number)
        
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
    
class QD_Upload():
    def __init__(self):
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
            r = requests.post(self.upload_url, files=files)
            r.close()
            result=json.loads(r.content)
            
            if result['success']:
                return result['fileUrl']
            else:
                warnings.warn('upload {} failed'.format(filename))
                return ''
            
        return ''
    
def save_and_upload(image_names,save_video_name,manager_dict):
	"""
	save the image in video and upload it
	"""
	codec = cv2.VideoWriter_fourcc(*"mp4v")
	writer = None
	
	for f in images_names:
		img=cv2.imread(f)
		if writer is None:
            height,width=img.shape[0:2]
			writer=cv2.VideoWriter(save_video_name,
							codec, fps,
							(width, height))
		
		writer.write(img)
		
	writer.release()
	
	loader=QD_Upload()
	manager_dict['fileUrl']=loader.upload(save_video_name)
    
if __name__ == '__main__':
    with open('config.json','r') as f:
        config=json.load(f)
    
    config['save_frame_number']=10
    p=QD_Process(config)
    p.process()
