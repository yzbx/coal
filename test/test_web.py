# -*- coding: utf-8 -*-

import unittest
import json
import time
import os

class WebTest(unittest.TestCase):  
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
        url=config['upload_url']

        with open('test.png','rb') as f:
            files = {'upload': f}
            r = requests.post(url, files=files)
            print('upload','*'*30)
            print(r.content)
            r.close()
            f.close()

        self.assertTrue(r.status_code==200 and r.ok)
        
if __name__ == '__main__':
    unittest.main()