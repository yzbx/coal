from flask import (Flask, render_template,
stream_with_context, Response, request,
make_response)
import time
import json
import argparse
import cv2
from app.bg_process import car_detection
flask_app = Flask(__name__)
g_rtsp_url='rtsp://admin:juancheng1@221.1.215.254:554'
g_car_detection=car_detection(g_rtsp_url)

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

@flask_app.route('/')
def demo():
    date=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    others={'date':date}
    return render_template('demo.html',
                            title='demo',
                            video_url='rtsp://admin:juancheng1@221.1.215.254:554',
                            task_name='detection_car',
                            others=json.dumps(others))

@flask_app.route('/helmet',methods=['POST','GET'])
def helmet():
    box=[[10,10,50,50,'helmet',0.9],[20,20,60,60,'none',0.8]]
    data=[]
    for x1,y1,x2,y2,cls,conf in box:
        d={'x1':x1,'x2':x2,'y1':y1,'y2':y2,'cls':cls,'conf':conf}
        data.append(d)
    return json.dumps({'bbox':data})

def gen_imencode(gen):
    for img in gen:
        ret, img = cv2.imencode('.jpg', img)
        frame=img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@flask_app.route('/start_demo',methods=['POST','GET'])
def start_demo():
    data={'video_url':None,'task_name':None,'others':None}
    for key in data.keys():
        flag,value=get_data(request,key)
        if not flag:
            return json.dumps(generate_error(1,'cannot obtain data {}'.format(key)))
        else:
            data[key]=value

    g_car_detection.update_video_url(data['video_url'])
    print('update video url')
    return Response(stream_with_context(gen_imencode(g_car_detection.process())),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-p','--port',help='port for web application',default=5005,type=int)
    args=parser.parse_args()
    flask_app.run(debug=True, host='0.0.0.0', port=args.port)
