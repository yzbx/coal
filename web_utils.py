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

def detection(data):
    if data['task_name']=='car_detection':
        with open('config.json','r') as f:
            config=json.load(f)

        config['video_url']=data['video_url']
        config['task_name']=data['task_name']
        
        try:
            others=json.loads(data['others'])
            for key,value in others.items():
                config['others'][key]=value
            logging.info('update others {}'.format(others))
        except:
            logging.warn('bad others format {}'.format(data['others']))

        try:
            p=QD_Process(config)
            p.process()
        except Exception as e:
            raise Exception('cannot start task because {}'.format(e.__str__()))
    else:
        raise Exception('no such task name')
    
    return 0

def kill_all_subprocess(root_pid=None):
    """
    kill all child process for root_pid
    if root_pid is not None:
        kill root_pid
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
            
    p = psutil.Process(root_pid)
    childs=p.children()
    for c in childs:
        if c.status=='zombie':
            logging.info('wait zombie pid={}'.format(c.pid))
            pid,status=os.waitpid(c.pid,os.WNOHANG)
        else:
            kill_group(c.pid)

    if root_pid is None:
        if p.children():
            logging.info('wait pid={}'.format(p.pid))
            os.wait()
            logging.info('wait pid={}'.format(p.pid))
    else:
        p.kill()
        logging.info('wait pid={}'.format(p.pid))
        os.wait()
        logging.info('wait pid={}'.format(p.pid))
    torch.cuda.empty_cache()