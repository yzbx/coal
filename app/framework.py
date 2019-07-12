import os
import cv2
import json
import time
import datetime
from easydict import EasyDict as edict
import requests
from multiprocessing import Process, Queue
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import Table,MetaData,create_engine,func
from sqlalchemy.orm import sessionmaker,Session
import warnings
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
        self.retry_times=0
        self.cap=None
        
        # queue for frame
        self.queue=None
        self.sub_process=None
        
    def read(self):
        """
        return flag and frame with VideoCapture restart
        flag=True when obtain valid frame (restart if neccessary)
        otherwise return flag=False (give up restart and exit programe)
        """
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
        assert self.sub_process is None
        if self.cap is not None:
            self.cap.release()
        self.video_url=video_url
        self.cap=cv2.VideoCapture(self.video_url)
        
    def read_from_queue(self):
        """
        skip old frames, always use the newest frame with subprocess
        """
        def write_to_queue(reader,q):
            while True:
                flag,frame=reader.read()
                time.sleep(0.01)
                if flag:
                    if q.qsize()<3:
                        q.put(frame)
                    else:
                        # two process get will cause error(get empty queue with q.get_nowait())
                        # or dead lock(get empty queue with q.get() with if q.qsize() is empty() )
                        q.get()
                        q.put(frame)
                else:
                    q.put(None)
                    break
        
        if self.queue is None:
            self.queue=Queue()
            self.sub_process=Process(target=write_to_queue,args=(self,self.queue))
            self.sub_process.start()
        
        while self.queue.qsize()>=3:
            frame=self.queue.get()
        else:
            # if queue is empty, it will wait here
            frame=self.queue.get()
         
        if frame is None:
            return False,frame
        else:
            return True,frame
        
    def join(self):
        if self.sub_process is not None:
            self.sub_process.join()
        
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
        self.queue=None
    
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
        self.queue=Queue()
        self.sub_process=Process(target=save_and_upload,args=(self.image_names,self.save_video_name,self.queue))
        self.sub_process.start()
#         self.sub_process.join() 
        
class QD_Detector(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        if hasattr(cfg,'task_name'):
            self.task_name=cfg.task_name
        else:
            self.task_name='car_detection'
        
        if hasattr(cfg,'others'):
            self.others=cfg.others
        else:
            self.others=''
        
        opt=edict()
        opt.cfg='app/config/yolov3.cfg'
        opt.data_cfg='app/config/coco.data'
        opt.weights='app/config/yolov3.weights'
        opt.img_size=416
        self.detector=yolov3_slideWindows(opt)
        
    def process(self,frame):
        image,bbox=self.detector.process_slide(frame)
        return image,bbox


class QD_Alerter(QD_Basic):
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
            # remove old image on disk
            for f in self.filenames[:-self.save_frame_number]:
                os.remove(f)
            self.filenames=self.filenames[-self.save_frame_number:]
        
        for idx,writer in enumerate(self.writers):
            if writer.can_upload():
                if writer.sub_process is None:
                    writer.upload_in_subprocess()
                elif writer.sub_process.is_alive():
                    print('{} is running'.format(writer.sub_process.pid))
                else:
                    # a sub process can join many times
                    writer.sub_process.join()
                    os.remove(writer.save_video_name)
                    self.writers[idx]=None
                    print('fileUrl is {fileUrl}'.format(fileUrl=writer.queue.get()))
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
            # flag,frame=self.reader.read()
            flag,frame=self.reader.read_from_queue()
            if flag:
                image,bbox=self.detector.process(frame)
                self.alerter.process(image,bbox)
            else:
                raise StopIteration('read frame failed!!!')
                break
    
    def demo(self):
        while True:
            # flag,frame=self.reader.read()
            flag,frame=self.reader.read_from_queue()
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
        self.upload_url=config['upload_url']
        
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

class QD_Database(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        Base = automap_base()
        self.engine = create_engine('mysql+pymysql://{user}:{passwd}@{host}:{port}/{database}'.format(
            user=self.cfg.user,
            passwd=self.cfg.passwd,
            host=self.cfg.host,
            port=self.cfg.port,
            database=self.cfg.database),echo=False,encoding="utf-8")
        
        Base.prepare(self.engine,reflect=True)
        self.Mtrp_Alarm=Base.classes.mtrp_alarm
        self.Mtrp_Alarm_Type=Base.classes.mtrp_alarm_type
        self.session=Session(self.engine)
    
    def __exit__(self):
        self.session.close()
        
    def insert(self,content):
        alarm=self.Mtrp_Alarm()
        alarm.alarmTime=datetime.datetime.now()
        alarm.content=content
        alarm.fileUrl=''
        
        alarm.createTime=datetime.datetime.now()
        
        max_id = self.session.query(func.max(self.Mtrp_Alarm.id)).scalar()
        if max_id is None:
            max_id=0
        alarm.id=max_id+1
        self.session.add(alarm)
        self.session.commit()
        return alarm.id
    
    def update(self,id,fileUrl):
        alarm=self.Mtrp_Alarm()
        self.session.query(self.Mtrp_Alarm).filter_by(id=id).update({'fileUrl':fileUrl})
        self.session.commit()
        
    def query(self,id):
        q=self.session.query(self.Mtrp_Alarm)
        result=q.filter(self.Mtrp_Alarm.id==id).one()
        return result
        
def save_and_upload(image_names,save_video_name,manager_dict):
    """
    save the image in video and upload it
    """
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    fps=30
    writer = None
    
    for f in image_names:
        img=cv2.imread(f)
        if writer is None:
            height,width=img.shape[0:2]
            writer=cv2.VideoWriter(save_video_name,
                            codec, fps,
                            (width, height))
        
        writer.write(img)
        
    writer.release()
    
    if not os.path.exists(save_video_name):
        raise Exception('cannot save images to {video}'.format(video=save_video_name))
        manager_dict.put('')
    else:
        loader=QD_Upload()
        fileUrl=loader.upload(save_video_name)
        manager_dict.put(fileUrl)
    
if __name__ == '__main__':
    with open('config.json','r') as f:
        config=json.load(f)
    
    config['save_frame_number']=10
    p=QD_Process(config)
    p.process()
