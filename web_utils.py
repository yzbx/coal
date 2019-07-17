import time
import json
import cv2
import os
from app.yolov3 import yolov3_detect
from app.framework import QD_Process
import psutil
import torch
import sys
import signal
import warnings
import logging

def generate_response(code,app_name,video_url,error_string='',succeed=0,pid=None):
    if pid is None:
        return {'succeed':succeed,'app_name':app_name,'error_code':code,
        'video_url':video_url,'error_string':error_string}
    else:
        return {'succeed':succeed,'app_name':app_name,'error_code':code,
        'video_url':video_url,'error_string':error_string,'pid':pid}

def get_data(request,name):
    if request.method == 'POST':
        value=request.form[name]
    elif request.method == 'GET':
        value=request.args.get(name)
    else:
        return False,None

    return True,value

def detection(data_json):
    data=json.loads(data_json)
    if data['task_name']=='car_detection':
        with open('config.json','r') as f:
            config=json.load(f)

        config['video_url']=data['video_url']
        config['task_name']=data['task_name']
        
        if isinstance(data['others'],dict):
            for key,value in data['others'].items():
                config['others'][key]=value
        else:
            warnings.warn('bad others format {}'.format(data['others']))
            
        try:
            p=QD_Process(config)
            p.process()
        except Exception as e:
            raise Exception('cannot start task because {}'.format(e.__str__()))
    else:
        raise Exception('no such task name')
    
    return 0

def kill_all_subprocess(pid=None):
    """
    kill all child process pids
    """
    
    def kill_group(pid):
        """
        kill pid and it's child
        """
        p=psutil.Process(pid)
        childs=p.children()
        for c in childs:
            kill_group(c.pid)
        
        p.kill()
            
    p = psutil.Process(pid)
    childs=p.children()
    for c in childs:
        if c.status=='zombie':
            pid,status=os.waitpid(c.pid,os.WNOHANG)
            logging.info('wait zombie pid={}'.format(pid))
        else:
            kill_group(c.pid)
    
    if p.children():   
        os.wait()
        
    torch.cuda.empty_cache()