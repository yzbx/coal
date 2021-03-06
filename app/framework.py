import os
import cv2
import json
import time
import datetime
from easydict import EasyDict as edict
import requests
import subprocess
from multiprocessing import Process, Queue
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine,func
from sqlalchemy.orm import Session
import numpy as np
import torch
import sys
if '.' not in sys.path:
    sys.path.insert(0,'.')
sys.path.insert(0,'./model/yolov3')
from app.algorithm import yolov3_slideWindows,vgg_fire
import logging
import redis

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
        # max_retry_times=-1 for unlimited retry times
        self.max_retry_times=-1
        self.retry_times=0
        self.cap=None

        # queue for frame
        self.queue=None
        self.sub_process=None
        self.time_out=3
        self.time_used=0

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
            self.time_used=0
            return True,frame
        else:
            logging.info('restart video capture {}'.format(self.video_url))
            self.cap.release()

            self.time_used+=0.5
            time.sleep(0.5)
            self.cap=cv2.VideoCapture(self.video_url)
            self.retry_times+=1
            if self.retry_times>self.max_retry_times>=0:
                raise StopIteration('retry times > {} for {}'.format(self.max_retry_times,self.video_url))
                return False,None

            if self.time_used>self.time_out:
                warn_img=np.zeros((600,800,3),dtype=np.uint8)
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                warn_img=cv2.putText(warn_img,text=self.video_url,org=(300,0),fontFace=fontFace,fontScale=2,color=(255,0,0),thickness=2)
                self.time_used=0
                logging.warning('use warning image for bad rtsp {}'.format(self.video_url))
                return True,warn_img
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
            self.sub_process.terminate()
            self.sub_process.join()

    def __del__(self):
        if self.sub_process is not None:
            self.sub_process.terminate()
            self.sub_process.join()

class QD_Writer(QD_Basic):
    """
    cannot save all frame in memory
    5min, 30fps 1024x2048 RGB image need memory:
    5*60*30*1024x2048x3 > 30G
    """
    def __init__(self,cfg,rule,filenames):
        super().__init__(cfg)
        self.rule=rule
        self.save_video_name=os.path.join('static',str(time.time())+'.mp4')
        self.convert_video_name=os.path.join('static','x264_'+os.path.basename(self.save_video_name))
        self.save_frame_number=self.cfg.save_frame_number

        valid_num=self.save_frame_number//2
        self.image_names=filenames[-valid_num:]
        self.sub_process=None
        self.queue=None

        # insert record to database
        self.database=QD_Database(cfg)

    def insert_database(self,content):
        self.id=self.database.insert(content)
        return self.id

    def update_database(self,fileUrl):
        self.database.update(self.id,fileUrl)

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

    def __del__(self):
        if self.sub_process is not None:
            self.sub_process.terminate()
            self.sub_process.join()

class QD_Detector(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)

        if hasattr(cfg,'others'):
            self.others=cfg.others
        else:
            self.others=''

        try:
            opt=self.get_opt()
            if cfg['task_name'] == 'fire':
                self.detector=vgg_fire(opt)
            else:
                self.detector=yolov3_slideWindows(opt)
        except Exception as e:
            self.detector=None
            raise Exception(e.__str__())


    def get_opt(self):
        if hasattr(self.cfg,'task_name'):
            task_name=self.cfg.task_name
        else:
            task_name='car'

        opt=edict()
        if task_name.startswith('test'):
            pass
        elif task_name.find('person')>=0:
            task_name='person'
        elif task_name.find('car')>=0:
            task_name='car'
        elif task_name.find('excavator')>=0:
            task_name='excavator'
        elif task_name.find('truck')>=0:
            task_name='truck'
        elif task_name.find('helmet_color')>=0:
            task_name='helmet_color'
        elif task_name.find('helmet')>=0:
            task_name='helmet'
        elif task_name=='fire':
            task_name='fire'
        else:
            logging.warning('unknown task name {}'.format(task_name))
            raise Exception('unknwn task name {}'.format(task_name))

        for model in self.cfg.models:
            if model.task_name == task_name:
                opt.cfg=model.cfg
                opt.data_cfg=model.data_cfg
                opt.weights=model.weights
                opt.img_size=416

                return opt

        raise Exception('cannot find model with task_name={}'.format(task_name))
        return opt

    def process(self,frame):
        #BUG process_slide has bug
        # image,bbox=self.detector.process_slide(frame)
        image,bbox=self.detector.process(frame)
        return image,bbox

    def __del__(self):
        del self.detector
        torch.cuda.empty_cache()


class QD_Alerter(QD_Basic):
    """
    save image to disk and record the filename
    """
    def __init__(self,cfg):
        super().__init__(cfg)
        self.filenames=[]
        self.writers=[]
        self.save_frame_number=self.cfg.save_frame_number
        self.max_filesize=self.save_frame_number*2
        self.cooling_time=0

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
        filename=os.path.join('static',str(time.time())+'.jpg')
        os.makedirs(os.path.dirname(filename),exist_ok=True)
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
                    pass
                else:
                    # a sub process can join many times
                    writer.sub_process.join()
                    os.remove(writer.save_video_name)
                    os.remove(writer.convert_video_name)
                    self.writers[idx]=None
                    fileUrl=writer.queue.get()
                    writer.update_database(fileUrl)
                    logging.info('update database fileUrl is {fileUrl}'.format(fileUrl=os.path.join(self.cfg.view_url,fileUrl)))
            else:
                writer.write_sync(filename)

        self.writers=[w for w in self.writers if w is not None]
        rule=self.bbox2rule(bbox)
        if self.cooling_time>0:
            self.cooling_time-=1
        if rule and self.cooling_time<=0:
            self.cooling_time=self.save_frame_number//2
            writer=QD_Writer(self.cfg,rule,self.filenames)
            writer.insert_database({'content':rule,'bbox':json.dumps(bbox)})
            logging.info('insert database rule is {}'.format(rule))
            self.writers.append(writer)

class QD_Process(QD_Basic):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.queue=None
        self.sub_process=None

        password=None if cfg['redis']['password']=="" else cfg['redis']['password']
        self.redis=redis.Redis(host=cfg['redis']['host'],port=cfg['redis']['port'],password=password)
        try:
            self.reader=QD_Reader(self.cfg.video_url)
            self.detector=QD_Detector(self.cfg)
            self.alerter=QD_Alerter(cfg)
        except Exception as e:
            raise Exception(e.__str__())

        logging.info(json.dumps(cfg))

    def save_to_redis(self,img,bbox):
        #pid=psutil.Process().pid
        key=self.cfg.redis_key
        retval, buffer = cv2.imencode('.jpg', img)
        img_bytes = np.array(buffer).tostring()
        result={'bbox':bbox,'img_bytes':img_bytes}
        # insert result to redis list with name key
        self.redis.lpush(key,json.dumps(result))
        # limit redis list with name key's size to 3
        self.redis.ltrim(key,0,3)

    def process(self):
        while True:
            # flag,frame=self.reader.read()
            flag,frame=self.reader.read_from_queue()
            if flag:
                image,bbox=self.detector.process(frame)
                self.save_to_redis(image,bbox)
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

    def __del__(self):
        if self.sub_process is not None:
            self.sub_process.terminate()
            self.sub_process.join()

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
#        with open(filename,'rb') as f:
#            files = {'upload': f}
#            r = requests.post(self.upload_url, files=files)
#            r.close()
#            result=json.loads(r.content)
#
#            if result['success']:
#                return result['fileUrl']
#            else:
#                logging.warning('upload {} failed'.format(filename))
#                print(result)
#                return ''

        try:
            headers = {"swagger-token":"yingji1"}
            with open(filename, 'rb') as f:
                files = {
                    'file': f,
                    #'deviceId': (None, str(others['deviceId'])),
                    #'channelId': (None, str(others['channelId'])),
                    #'algoId': (None, str(others['algoId'])),
                }
                r = requests.post(self.upload_url, files=files, headers=headers)
                r.close()
                result = json.loads(r.content.decode())

                if not result['success']:
                    logging.warning('upload image {} failed'.format(filename))

                return result['data']['path']
        except Exception as e:
            logging.warning(e.__str__())
            return 'upload failed!'



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
        #self.Mtrp_Alarm_Type=Base.classes.mtrp_alarm_type
        self.session=Session(self.engine)

    def __exit__(self):
        self.session.close()

    def insert(self,data_dict,event_id=1):
        alarm=self.Mtrp_Alarm()
        alarm.alarmTime=datetime.datetime.now()
        alarm.content=data_dict['content']
        alarm.bbox=data_dict['bbox']
        alarm.fileUrl=''
        alarm.imgPath=''
        alarm.event_id=event_id
        alarm.logID=0
        alarm.alarm_type=self.cfg.task_name
        alarm.device_id=self.cfg.others.device_id
        alarm.channel_no=self.cfg.others.channel_no
        alarm.createTime=datetime.datetime.now()

        max_id = self.session.query(func.max(self.Mtrp_Alarm.id)).scalar()
        if max_id is None:
            max_id=0
        alarm.id=max_id+1
        self.session.add(alarm)
        self.session.commit()
        return alarm.id

    def update(self,id,fileUrl):
        videoPath,imgPath=fileUrl
        self.session.query(self.Mtrp_Alarm).filter_by(id=id).update({'fileUrl':videoPath,'imgPath':imgPath})
        self.session.commit()

    def query(self,id):
        q=self.session.query(self.Mtrp_Alarm)
        result=q.filter(self.Mtrp_Alarm.id==id).one()
        return result

def save_and_upload(image_names,save_video_name,queue,upload=True):
    """
    save the image in video and upload it
    """
    with open('config.json','r') as f:
        config=json.load(f)

    codec = cv2.VideoWriter_fourcc(*"mp4v")
#    codec = cv2.VideoWriter_fourcc(*'X264')
#    codec=0x21
#    codec = cv2.VideoWriter_fourcc(*'CJPG')
#    codec = cv2.VideoWriter_fourcc(*'avc1')
#    codec = cv2.VideoWriter_fourcc(*'MP4V')
#    codec = cv2.VideoWriter_fourcc(*'H264')
    fps=config['save_frame_rate']
    writer = None

    for f in image_names:
        img=cv2.imread(f)
        if writer is None:
            height,width=img.shape[0:2]
            os.makedirs(os.path.dirname(save_video_name),exist_ok=True)
            writer=cv2.VideoWriter(save_video_name,
                            codec, fps,
                            (width, height))

        writer.write(img)

    writer.release()
    logging.info('write {} images with fps={} into video'.format(len(image_names),fps))

    convert_video_name=os.path.join('static','x264_'+os.path.basename(save_video_name))
    convert_cmd='ffmpeg -i {} -vcodec h264 {}'.format(save_video_name,convert_video_name)
    subprocess.run(convert_cmd,shell=True)

    if not os.path.exists(save_video_name):
        logging.warning('cannot save images to {video}'.format(video=save_video_name))
        raise Exception('cannot save images to {video}'.format(video=save_video_name))
        queue.put(('',''))

    if not os.path.exists(convert_video_name):
        logging.warning('cannot convert {in_video} to {out_video}'.format(in_video=save_video_name,out_video=convert_video_name))
        raise Exception('cannot convert {in_video} to {out_video}'.format(in_video=save_video_name,out_video=convert_video_name))
        queue.put(('',''))

    if upload:
        loader=QD_Upload()
        videoPath=loader.upload(convert_video_name)
        imgPath=loader.upload(image_names[0])
        queue.put((videoPath,imgPath))

if __name__ == '__main__':
    with open('config.json','r') as f:
        config=json.load(f)

    config['save_frame_number']=10
    p=QD_Process(config)
    p.process()
