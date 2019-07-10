import cv2
import json
import time
from app.framework import QD_Reader
import matplotlib.pyplot as plt
from multiprocessing import Queue,Process

def test_rtsp():
    with open('config.json','r') as f:
        config=json.load(f)
    video_url=config['video_url']
    reader=QD_Reader(video_url)
    
    plt.ion()
    plt.show()
    for i in range(5):
        flag,frame=reader.read()
        if flag:
            plt.imshow(frame)
            plt.pause(0.001)
        time.sleep(0.5)
        print('read frame',i)
        
def worker_newest(q,reader):
    for i in range(1000):
        flag,frame=reader.read()
        time.sleep(0.01)
        if flag:
            if q.qsize()<3:
                q.put(frame)
            else:
                # two process get will cause error(get empty queue with q.get_nowait())
                # or dead lock(get empty queue with q.get())
                q.get()
                q.put(frame)
    q.put(None)
        
def worker_all(q,reader):
    for i in range(1000):
        flag,frame=reader.read()
        time.sleep(0.01)
        if flag:
            q.put(frame)
    q.put(None)
    
def test_rtsp_queue(worker):
    with open('config.json','r') as f:
        config=json.load(f)
    video_url=config['video_url']
    reader=QD_Reader(video_url)
        
    q=Queue()
    p=Process(target=worker,args=(q,reader))
    p.start()
    
    plt.ion()
    plt.show()
    idx=0
    while True:
        while not q.empty():
            frame=q.get()
            # need this condition, otherwise stucked!!!
            if q.qsize()<3:
                break
        else:
            continue
            
        idx=idx+1
        print('get frame',idx)
        time.sleep(1)
        if frame is not None:
            plt.imshow(frame)
            plt.pause(0.001)
        else:
            break
        
    p.join()
    
if __name__ == '__main__':
#     print('test rtsp')
#     test_rtsp()
    print('test rtsp queue')
    test_rtsp_queue(worker_newest)