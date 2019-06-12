from flask import Flask
from utils import detection
from flask import request, jsonify
from flask import render_template,redirect,url_for
import subprocess
import multiprocessing
import psutil
import time
import json
import argparse

flask_app = Flask(__name__)
app_config=[]

@flask_app.route('/')
def index():
    return render_template('result.html',
                            title='test',
                            pid=0,
                            status='?',
                            is_running='?',
                            video_url='rtsp/xxx',
                            task_name='detection',
                            others='2019/05/28')

def generate_error(code,str='',succeed=0,pid=None):
    if pid is None:
        return {'succeed':succeed,'error_code':code,
        'error_string':str}
    else:
        return {'succeed':succeed,'error_code':code,
        'error_string':str,'pid':pid}

def get_data(request,name):
    if request.method == 'POST':
        value=request.form[name]
    elif request.method == 'GET':
        value=request.args.get(name)
    else:
        return False,0

    return True,value

def get_app_id(data):
    for cfg in app_config:
        if data['video_url']==cfg['video_url'] and \
            data['task_name']==cfg['task_name']:
            return cfg['pid']
    
    return -1

@flask_app.route('/longtask', methods=['POST', 'GET'])
def longtask():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_error(1,'cannot obtain data {}'.format(key)))
        else:
            data[key]=value

    proc=multiprocessing.Process(target=detection,args=[json.dumps(data)])
    proc.start()
    data['pid']=proc.pid
    assert data['pid']>0
    app_config.append(data)
    return json.dumps(generate_error(0,succeed=1,pid=proc.pid))

@flask_app.route('/taskresult/<pid>')
def taskresult(pid):
    try:
        p = psutil.Process(int(pid))
    except Exception as e:
        return json.dumps(generate_error(3,succeed=1,pid=pid,str=e.__str__()))
    
    return render_template('result.html',
                            title=pid,
                            pid=pid,
                            status=p.status(),
                            is_running=p.is_running(),
                            video_url='rtsp/xxx',
                            task_name='detection',
                            others='2019/05/28')

@flask_app.route('/stoptask',methods=['POST','GET'])
def stoptask():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_error(1,'cannot obtain data {}'.format(key)))
        else:
            data[key]=value

    pid=get_app_id(data) 
    if pid==-1:
        return json.dumps(generate_error(2,'no process running for {}/{}'.format(data['video_url'],data['task_name'])))

    try:
        p = psutil.Process(pid)
        p.terminal()
    except Exception as e:
        return json.dumps(generate_error(3,succeed=1,pid=pid,str=e.__str__()))
    return  json.dumps(generate_error(0,succeed=1,pid=pid))
        
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--port',help='port for web application',default=5005,type=int)
    args=parser.parse_args()
    flask_app.run(debug=True, host='0.0.0.0', port=args.port)
