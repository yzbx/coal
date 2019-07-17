import time
import json
import cv2
import os
from app.framework import QD_Process
import psutil
import torch
import sys
import signal
import warnings
import logging
from multiprocessing import Queue,Process

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

def detection_demo(data):
    def write_to_queue(config,q):
        worker=QD_Process(config)
        while True:
            flag,frame=worker.reader.read_from_queue()
            if flag:
                image,bbox=worker.detector.process(frame)
                q.put(image)
            else:
                q.put(None)
                break
    
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
            queue=Queue()
            sub_process=Process(target=write_to_queue,args=(config,queue))
            sub_process.start()
            
            while True:
                image=queue.get()
                if image is not None:
                    yield image
                else:
                    logging.warn('terminate subprocess')
                    sub_process.terminate()
                    logging.warn('join subprocess')
                    sub_process.join()
                    logging.warn('raise exception')
                    raise StopIteration('not image offered')
                    
        except Exception as e:
            raise Exception('cannot start task because {}'.format(e.__str__()))
    else:
        raise Exception('no such task name')

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