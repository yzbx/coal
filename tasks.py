#!/usr/bin/env python

from flask import Flask,Response,stream_with_context
from web_utils import detection,generate_response,get_data,kill_all_subprocess
from flask import request, jsonify, flash, send_from_directory
from flask import render_template,redirect,url_for
import subprocess
import multiprocessing
import psutil
import time
import json
import argparse
import os
import requests
import os
import sys
import signal
from app.app_utils import gen_imencode,check_rtsp,get_status
from app.framework import QD_Process
flask_app = Flask(__name__)

# record video_url, task_name and pid
app_config=[]

def get_app_id(data):
    for cfg in app_config:
        if data['video_url']==cfg['video_url'] and \
            data['task_name']==cfg['task_name']:
            return cfg['pid']

    return -1

@flask_app.route('/')
def index():
    date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return render_template('index.html',
                            title='index',
                            pid=0,
                            status='?',
                            is_running='?',
                            video_url='rtsp://admin:juancheng1@221.1.215.254:554',
                            task_name='detection_car',
                            others='{"date":"%s"}'%date)

@flask_app.route('/status')
def status():
    return get_status()

@flask_app.route('/kill')
def kill_subprocess():
    kill_all_subprocess()
    return "kill all sub process"

@flask_app.route('/error',methods=['POST','GET'])
def error():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_response(1,
                                             app_name='start_demo',
                                             video_url=data['video_url'],
                                             error_string='cannot obtain data {}'.format(key)))
        else:
            data[key]=value
    
#    os.execv(__file__, sys.argv)
    error_string="pid={pid} \n out of gpu memory".format(pid=os.getpid())
    return json.dumps(generate_response(2,
                                  app_name='start_demo',
                                  video_url=data['video_url'],
                                  error_string=error_string))

@flask_app.route('/restart',methods=['POST','GET'])
def restart():
    """
    just restart demo, kill demo pid
    """
    data={'flag':'False'}
    for key in data.keys():
        flag,value=get_data(request,key)
        if flag:
            data[key]=value
    if data['flag']=='True':
        kill_all_subprocess()
        return "restart okay"
    else:
        return "hello world"

@flask_app.route('/start_task', methods=['POST', 'GET'])
def start_task():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_response(1,
                                             app_name='start_task',
                                             video_url=data['video_url'],
                                             error_string='cannot obtain data {}'.format(key)))
        else:
            data[key]=value
    
    print(data)
    if not check_rtsp(data['video_url']):
        return json.dumps(generate_response(3,
                                  app_name='start_demo',
                                  video_url=data['video_url'],
                                  error_string="cannot open rtsp"))
    
    pid=get_app_id(data)
    if pid!=-1:
        return json.dumps(generate_response(2,video_url=data['video_url'],
                                         app_name='start_task',
                                         error_string='already has process running for {}/{}'.format(data['video_url'],data['task_name'])))
        
    proc=multiprocessing.Process(target=detection,args=[json.dumps(data)])
    proc.start()
    data['pid']=proc.pid
    assert data['pid']>0
    app_config.append(data)
    return json.dumps(generate_response(0,succeed=1,pid=proc.pid,
                                     app_name='start_task',
                                     video_url=data['video_url']))

@flask_app.route('/task_result/<pid>')
def task_result(pid):
    try:
        p = psutil.Process(int(pid))
    except Exception as e:
        return json.dumps(generate_response(3,
                                         video_url='',
                                         app_name='task_result',
                                         succeed=1,pid=pid,
                                         error_string=e.__str__()))

    return render_template('result.html',
                            title=pid,
                            pid=pid,
                            status=p.status(),
                            is_running=p.is_running(),
                            video_url='rtsp/xxx',
                            task_name='detection',
                            others='2019/05/28')

@flask_app.route('/stop_task',methods=['POST','GET'])
def stop_task():
    data={'video_url':None,'task_name':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_response(1,
                                             video_url=data['video_url'],
                                             app_name='stop_task',
                                             error_string='cannot obtain data {}'.format(key)))
        else:
            data[key]=value

    pid=get_app_id(data)
    if pid==-1:
        return json.dumps(generate_response(2,video_url=data['video_url'],
                                         app_name='stop_task',
                                         error_string='no process running for {}/{}'.format(data['video_url'],data['task_name'])))

    try:
        kill_all_subprocess(pid)
        os.kill(pid,signal.SIGKILL)
        os.wait()
    except Exception as e:
        return json.dumps(generate_response(3,
                                         video_url=data['video_url'],
                                         app_name='stop_task',
                                         succeed=1,pid=pid,error_string=e.__str__()))
    else:
        for d in app_config:
            if d['pid']==pid:
                app_config.remove(d)
        return  json.dumps(generate_response(0,
                                          video_url=data['video_url'],
                                          app_name='stop_task',
                                          succeed=1,
                                          pid=pid))


@flask_app.route('/demo')
def demo():
    date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return render_template('demo.html',
                            title='demo',
                            video_url='rtsp://admin:juancheng1@221.1.215.254:554',
                            task_name='detection_car',
                            others='{"date":"%s"}'%date)

@flask_app.route('/start_demo',methods=['POST','GET'])
def start_demo():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_response(1,
                                             app_name='start_demo',
                                             video_url=data['video_url'],
                                             error_string='cannot obtain data {}'.format(key)))
        else:
            data[key]=value
            
    if not check_rtsp(data['video_url']):
        return json.dumps(generate_response(3,
                                  app_name='start_demo',
                                  video_url=data['video_url'],
                                  error_string="cannot open rtsp"))

    try:
        with open('config.json','r') as f:
            config=json.load(f)
        config['video_url']=data['video_url']
        config['task_name']=data['task_name']
        config['others']=data['others']
        p=QD_Process(config)
    except RuntimeError as e:
        return json.dumps(generate_response(2,
                                  app_name='start_demo',
                                  video_url=data['video_url'],
                                  error_string=e.__str__()))
#        return redirect(url_for('error'),code=307)
    else:
        return Response(stream_with_context(gen_imencode(p.demo())),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--port',help='port for web application',default=5005,type=int)
    args=parser.parse_args()
    flask_app.run(debug=True, host='0.0.0.0', port=args.port)
