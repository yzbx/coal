# -*- coding: utf-8 -*-

import unittest
import json
import time
import os

class TestMethods(unittest.TestCase):
    def test_json(self):
        cfg={
            'host':"10.0.0.39",
            'user':"iscas",
            'passwd':"sketch_123",
            'port':8306,
            'database':'qingdao',
            'upload_url':'http://10.50.200.171:8080/mtrp/file/json/upload.jhtml',
            "video_url":"rtsp://admin:juancheng1@221.1.215.254:554",
            "task_name":"car_detection",
            "others":{"a":1,"b":2},
          }

        if not os.path.exists('config.json'):
            with open('config.json','r') as f:
                json.dump(cfg,f)
        self.assertTrue(True)

    def test_upload(self):
        import requests
        with open('config.json','r') as f:
            config=json.load(f)
        url=config.upload_url

        with open('test.png','rb') as f:
            files = {'upload': f}
            r = requests.post(url, files=files)
            print('upload','*'*30)
            print(r.content)
            r.close()
            f.close()

        self.assertTrue(r.status_code==200 and r.ok)

    def test_sqlalchemy(self):
        from sqlalchemy import create_engine
        from sqlalchemy.sql import select
        from database import metadata,mtrp_alarm,mtrp_alarm_type
#        engine = create_engine('sqlite:///sqlite.db', echo=True)
        with open('config.json','r') as f:
            config=json.load(f)

        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(config['user'],
                               config['passwd'],
                               config['host'],
                               config['port'],
                               config['database']),
                               echo=True)
        conn=engine.connect()
        # This will check for the presence of each table first before creating, so it’s safe to call multiple times:
        metadata.create_all(engine)

        print('select mtrp_alarm','*'*30)
        s=select([mtrp_alarm.c.id,mtrp_alarm.c.pic_name,mtrp_alarm.c.pic_url])
        result=conn.execute(s)
        for row in result:
            print(row)

        result.close()

        print('select mtrp_alarm_type','*'*30)
        s=select([mtrp_alarm_type.c.id])
        result=conn.execute(s)
        for row in result:
            print(row)

        result.close()
        self.assertTrue(True)

    def test_mysql_connector(self):
        import mysql.connector
        with open('config.json','r') as f:
            config=json.load(f)
            f.close()

        mydb = mysql.connector.connect(
          host=config['host'],
          user=config['user'],
          passwd=config['passwd'],
          port=config['port'],
        )
        print(mydb)
        mydb.close()
        self.assertTrue(True)
        
    def test_multi_process(self):
        from multiprocessing import Process
        def fun(note):
            for i in range(10):
                time.sleep(1)
                print("note: {}".format(note),i)
        
        
        p1=Process(target=fun,args=('detection car',))
        print('start p1')
        p1.start()
        p2=Process(target=fun,args=('detection helmet',))
        print('start p2')
        p2.start()
        print('join p1')
        p1.join()
        print('join p2')
        p2.join()
        
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()