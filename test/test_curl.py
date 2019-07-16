# -*- coding: utf-8 -*-

import subprocess
import unittest
import json

class CurlTest(unittest.TestCase):        
    def test_start(self):
        with open('config.json','r') as f:
            config=json.load(f)
        
        url='127.0.0.1:8205/start_task'
        cmd='curl -G -X GET --data "video_url={}&task_name={}&others=hello" {}'.format(config['video_url'],config['task_name'],url)
        
        print('\n')
        print(cmd)
        result=subprocess.check_output(cmd,shell=True).decode('utf-8')
        print(result)
        d=json.loads(result)
        self.assertTrue(d['succeed']==1 and d['video_url']==config['video_url'])
        
        result=subprocess.check_output(cmd,shell=True).decode('utf-8')
        print(result)
        d=json.loads(result)
        
        self.assertTrue(d['succeed']==0 and d['video_url']==config['video_url'])

    def test_stop(self):
        with open('config.json','r') as f:
            config=json.load(f)
        
        url='127.0.0.1:8205/stop_task'
        cmd='curl -G -X GET --data "video_url={}&task_name={}&others=hello" {}'.format(config['video_url'],config['task_name'],url)
        
        print('\n')
        print(cmd)
        result=subprocess.check_output(cmd,shell=True).decode('utf-8')
        print(result)
        d=json.loads(result)
        self.assertTrue(d['succeed']==1 and d['video_url']==config['video_url'])
        
        result=subprocess.check_output(cmd,shell=True).decode('utf-8')
        print(result)
        d=json.loads(result)
        
        self.assertTrue(d['succeed']==0 and d['video_url']==config['video_url'])
        
    def test_kill(self):
        url='127.0.0.1:8205/kill'
        cmd='curl {}'.format(url)
        result=subprocess.check_output(cmd,shell=True).decode('utf-8')
        print(result)
        self.assertTrue(result.find('restart')>=0)
        
if __name__ == '__main__':
    unittest.main()