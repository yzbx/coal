#!/usr/bin/env python

from flask import Flask,Response,stream_with_context
from web_utils import detection,detection_demo,generate_response,get_data,kill_all_subprocess
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
import logging

flask_app = Flask(__name__,static_url_path='/static')

logging.basicConfig(filename='qd.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d %(message)s')
# record video_url, task_name and pid
app_config=[]

def get_app_id(data):
    global app_config
    for cfg in app_config:
        if data['video_url']==cfg['video_url'] and \
            data['task_name']==cfg['task_name']:
            return cfg['pid']

    return -1

@flask_app.route('/')
def index():
    with open('config.json','r') as f:
        config=json.load(f)
    p=psutil.Process()
    return render_template('index.html',
                            title='index',
                            pid=p.pid,
                            status=p.status(),
                            is_running=p.is_running(),
                            video_url=config['video_url'],
                            task_name=config['task_name'],
                            others=config['others'])

#@flask_app.route('/file/<path:path>')
#def send_js(path):
#    return send_from_directory('test', path)

@flask_app.route('/qd.log')
def log():
    with open('qd.log','r') as f:
        content=f.read()
        return content.replace('\n','<br>').replace(' ','&nbsp')
    
@flask_app.route('/status')
def status():
    global app_config
    task_status=app_config.__str__()
    return get_status()+"<br>"+task_status

@flask_app.route('/kill')
def kill_subprocess():
    global app_config
    kill_all_subprocess()
    app_config=[]
    return redirect(url_for('status'))

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
    global app_config
    
    data={'flag':'False'}
    for key in data.keys():
        flag,value=get_data(request,key)
        if flag:
            data[key]=value
    if data['flag']=='True':
        kill_all_subprocess()
        app_config=[]
        return redirect(url_for('status'))
    else:
        return "hello world"

@flask_app.route('/start_task', methods=['POST', 'GET'])
def start_task():
    global app_config
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
    
    try:
        proc=multiprocessing.Process(target=detection,args=[data])
        proc.start()
        data['pid']=proc.pid
        assert data['pid']>0
        logging.info(app_config)
        app_config.append(data)
        logging.info(app_config)
        return json.dumps(generate_response(0,succeed=1,pid=proc.pid,
                                         app_name='start_task',
                                         video_url=data['video_url']))
    except Exception as e:
        return json.dumps(generate_response(2,video_url=data['video_url'],
                                         app_name='start_task',
                                         error_string=e.__str__()))

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
    
    result={}
    result['pid']=p.pid
    result['status']=p.status()
    result['is_running']=p.is_running()
    return json.dumps(result)

@flask_app.route('/stop_task',methods=['POST','GET'])
def stop_task():
    global app_config
    
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
    except Exception as e:
        return json.dumps(generate_response(3,
                                         video_url=data['video_url'],
                                         app_name='stop_task',
                                         succeed=1,
                                         pid=pid,
                                         error_string=e.__str__()))
    else:
        logging.info(app_config)
        for d in app_config:
            if d['pid']==pid:
                app_config.remove(d)
        logging.info(app_config)
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
        p=detection_demo(config)
    except RuntimeError as e:
        return json.dumps(generate_response(2,
                                  app_name='start_demo',
                                  video_url=data['video_url'],
                                  error_string=e.__str__()))
#        return redirect(url_for('error'),code=307)
    else:
        return Response(stream_with_context(gen_imencode(p)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--port',help='port for web application',default=5005,type=int)
    args=parser.parse_args()
    flask_app.run(debug=True, host='0.0.0.0', port=args.port)
