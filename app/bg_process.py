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

        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(config['user'],
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
        self.count=0

    def add(self,bbox):
        alarm=self.mtrp_alrm()
        alarm.alarmTime=int(time.time())
        alarm.content="find {} car".format(10)
        #todo
        alarm.fileUrl=''
        self.session.add(alarm)

    def commit(self):
        self.session.commit()


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

    def process(self):
        while True:
            flag,frame=self.reader.read()
            if flag:
                image,bbox=self.detector.process_slide(frame)
                yield image
