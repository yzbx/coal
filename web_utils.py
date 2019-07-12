import time
import json
import cv2
import os
from app.yolov3 import yolov3_detect
from app.framework import QD_Process
import psutil

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
        config['others']=data['others']
        try:
            p=QD_Process(config)
            p.process()
        except Exception as e:
            raise Exception('cannot start task because {}'.format(e.__str__()))
    else:
        raise Exception('no such task name')

    return 0

def kill_all_subprocess(pids=None):
    """
    if pids==None: kill all child process pids
    else:kill pids
    """
#    current_process = psutil.Process()
#    children = current_process.children(recursive=True)
#    for child in children:
#        print('Child pid is {}'.format(child.pid))
    def on_terminate(proc):
        print("process {} terminated with exit code {}".format(proc, proc.returncode))
    
    procs = psutil.Process().children()
    if pids is not None:
        procs = [p for p in procs if p.pid in pids]
        
    if len(procs)>0:
        for p in procs:
            p.terminate()
        gone, alive = psutil.wait_procs(procs, timeout=3, callback=on_terminate)
        for p in alive:
            p.kill()