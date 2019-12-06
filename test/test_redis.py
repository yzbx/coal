# -*- coding: utf-8 -*-
"""
reference:
    1. https://gist.github.com/gachiemchiep/52f3255a81c907461c2c7ced6ede367a
    2. https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
    3. https://github.com/nicolasff/webdis
    4. https://www.runoob.com/redis/redis-security.html

"""
import unittest
import json
import time
import os
import redis
import cv2
import numpy as np

class RedisTest(unittest.TestCase):
    def test_redis(self):
        with open('config.json','r') as f:
            config=json.load(f)

        password=None if config['redis']['password']=="" else config['redis']['password']
        r=redis.Redis(host=config['redis']['host'],port=config['redis']['port'],password=password)
        img_path='test.png'
        img=cv2.imread(img_path)
        retval, buffer = cv2.imencode('.png', img)
        img_bytes = np.array(buffer).tostring()
        r.set(img_path,img_bytes)

        fetch_img_bytes=r.get(img_path)
        decoded=cv2.imdecode(np.frombuffer(fetch_img_bytes,np.uint8),cv2.IMREAD_COLOR)
        self.assertTrue((img==decoded).all())

if __name__ == '__main__':
    unittest.main()