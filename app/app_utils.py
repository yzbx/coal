import cv2
import psutil
import subprocess
import json
def gen_imencode(gen):
    for img in gen:
        ret, img = cv2.imencode('.jpg', img)
        frame=img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def check_rtsp(video_url):
    cap=cv2.VideoCapture(video_url)
    flag=cap.isOpened()
    cap.release()
    return flag

def get_process_status():
    d={}
    current_process = psutil.Process()
    d['main']=current_process.pid
    children = current_process.children(recursive=False)
    child_pid=[[child.pid,child.status()] for child in children]
    d['child']=child_pid
    
    for pid,status in child_pid:
        child=psutil.Process(pid)
        grandchilds=child.children()
        d[str(pid)]=[[g.pid,g.status()] for g in grandchilds]
        
    return d

def get_status():
    d={}
    d['process']=get_process_status()
    d['nvidia-smi']=subprocess.check_output(['nvidia-smi']).decode('utf-8')
    
    html=''
    for k,v in d.items():
        html+='<h1>{}<h1><br>'.format(k)
        if isinstance(v,dict):
            html+=json.dumps(v).replace('\n','<br>').replace(' ','&nbsp')
        else:
            html+=v.replace('\n','<br>').replace(' ','&nbsp')
    return html