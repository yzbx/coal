from flask import Flask,Response,stream_with_context
from web_utils import detection,video_player,app_player
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

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'media'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

flask_app = Flask(__name__)
flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# record video_url, task_name and pid
app_config=[]

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

@flask_app.route('/demo')
def demo():
    date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return render_template('demo.html',
                            title='demo',
                            video_url='rtsp://admin:juancheng1@221.1.215.254:554',
                            task_name='detection_car',
                            others='{"date":"%s"}'%date)
    
@flask_app.route('/database')
def database():
    import mysql.connector
        
    with open('config.json','r') as f:
        config=json.load(f)
        f.close()
    mydb = mysql.connector.connect(
      host=config['host'],
      user=config['user'],
      passwd=config['passwd'],
      port=config['port'],
      database=config['database'],
    )

    mycursor = mydb.cursor()

    mycursor.execute("SHOW TABLES")
    print('table','*'*30)
    tables=[]
    for x in mycursor:
      print(x)
      tables.append(str(x))
    
    mycursor.close()
    return "tables: "+' '.join(tables)

def generate_error(code,app_name,video_url,error_string='',succeed=0,pid=None):
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

def get_app_id(data):
    for cfg in app_config:
        if data['video_url']==cfg['video_url'] and \
            data['task_name']==cfg['task_name']:
            return cfg['pid']
    
    return -1

@flask_app.route('/start_task', methods=['POST', 'GET'])
def start_task():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_error(1,
                                             app_name='start_task',
                                             video_url=data['video_url'],
                                             error_string='cannot obtain data {}'.format(key)))
        else:
            data[key]=value

    proc=multiprocessing.Process(target=detection,args=[json.dumps(data)])
    proc.start()
    data['pid']=proc.pid
    assert data['pid']>0
    app_config.append(data)
    return json.dumps(generate_error(0,succeed=1,pid=proc.pid,
                                     app_name='start_task',
                                     video_url=data['video_url']))

@flask_app.route('/task_result/<pid>')
def task_result(pid):
    try:
        p = psutil.Process(int(pid))
    except Exception as e:
        return json.dumps(generate_error(3,
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
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_error(1,
                                             video_url=data['video_url'],
                                             app_name='stop_task',
                                             error_string='cannot obtain data {}'.format(key)))
        else:
            data[key]=value

    pid=get_app_id(data) 
    if pid==-1:
        return json.dumps(generate_error(2,video_url=data['video_url'],
                                         app_name='stop_task',
                                         error_string='no process running for {}/{}'.format(data['video_url'],data['task_name'])))

    try:
        p = psutil.Process(pid)
        p.terminal()
    except Exception as e:
        return json.dumps(generate_error(3,
                                         video_url=data['video_url'],
                                         app_name='stop_task',
                                         succeed=1,pid=pid,error_string=e.__str__()))
    return  json.dumps(generate_error(0,
                                      video_url=data['video_url'],
                                      app_name='stop_task',
                                      succeed=1,
                                      pid=pid))
        

@flask_app.route('/start_demo',methods=['POST','GET'])
def start_demo():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_error(1,'cannot obtain data {}'.format(key)))
        else:
            data[key]=value
    
    return Response(stream_with_context(app_player(data['video_url']).gen()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@flask_app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(flask_app.config['UPLOAD_FOLDER'],exist_ok=True)
            file.save(os.path.join(flask_app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return ""

@flask_app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(flask_app.config['UPLOAD_FOLDER'],
                               filename)
    
@flask_app.route('/upload_to_server',methods=['POST','GET'])
def upload_to_server():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        url='http://10.50.200.171:8080/mtrp/file/json/upload.jhtml'
        files = {'file': file}
        r = requests.post(url, files=files)
        return r.content
        
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--port',help='port for web application',default=5005,type=int)
    args=parser.parse_args()
    flask_app.run(debug=True, host='0.0.0.0', port=args.port)
