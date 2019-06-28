"""
background process for multiprocessing
"""
import cv2
import time
import json
from easydict import EasyDict as edict
from sqlalchemy import Table,MetaData,create_engine
from sqlalchemy.orm import sessionmaker
from videowrite import yolov3_slideWindows,MyVideoCapture,MyVideoWriter

class mysql_orm():
    def __init__(self):
        metadata=MetaData()
        with open('config.json','r') as f:
            config=json.load(f)

        self.engine=engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(config['user'],
                               config['passwd'],
                               config['host'],
                               config['port'],
                               config['database']),
                               echo=True)

        self.mtrp_alarm=Table('mtrp_alarm',metadata,autoload=True,
        autoload_with=engine)
        self.mtrp_alarm_type=Table('mtrp_alarm',metadata,autoload=True,
        autoload_with=engine)

        conn=engine.connect()
        # This will check for the presence of each table first before creating, so it’s safe to call multiple times:
        metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)
        self.bbox_count={}
        self.alarm=None
        self.alarm_type=None

    def add(self,bbox,file_url):
        # save alarm to mysql dataset if file_url is not None
        if file_url is not None:
            self.alarm.content='find'
            for k,v in self.bbox_count.items():
                self.alarm.content+=" {} {}".format(v,k)
            self.alarm.fileUrl=file_url
            #todo
            self.session.add(alarm)
            self.session.new()
            self.session.commit()
            self.session.close()
            self.session=sessionmaker(bind=self.engine)
            self.alarm=None
            self.alarm_type=None

        # count for bbox
        if self.alarm is None:
            alarm=self.mtrp_alrm()
            #todo
            alarm.alarmTime=int(time.time())
            self.bbox_count={}

        for d in bbox:
            label=d['label']
            if label in self.bbox_count.keys():
                self.bbox_count[label]+=1
            else:
                self.bbox_count[label]=1


class car_detection():
    def __init__(self,video_url):
        self.video_url=video_url
        opt=edict()
        opt.cfg='app/config/yolov3.cfg'
        opt.data_cfg='app/config/coco.data'
        opt.weights='app/config/yolov3.weights'
        opt.img_size=416

        self.reader=MyVideoCapture(video_url)
        self.writer=MyVideoWriter()
        self.detector=yolov3_slideWindows(opt)
        self.mysql=mysql_orm()
        self.time=time.time()

        self.status="running"

    def process(self):
        while True:
            flag,frame=self.reader.read()
            if flag:
                image,bbox=self.detector.process_slide(frame)
                yield image

    def bg_process(self):
        while self.status=='running':
            flag,frame=self.reader.read()
            if flag:
                image,bbox=self.detector.process_slide(frame)
                file_url=self.writer.write(image)
                #self.mysql.add(bbox,file_url)

    def stop(self):
        self.status='stop'
