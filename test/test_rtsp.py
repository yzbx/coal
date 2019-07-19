import json
import time
from app.framework import QD_Reader,save_and_upload
import matplotlib.pyplot as plt
from multiprocessing import Queue,Process
import cv2
import os 
import warnings

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
    
def test_read_from_queue():
    with open('config.json','r') as f:
        config=json.load(f)
    video_url=config['video_url']
    reader=QD_Reader(video_url)
    
    plt.ion()
    plt.show()
    idx=0
    while True:
        time.sleep(1)
        idx=idx+1
        flag,frame=reader.read_from_queue()
        if frame is not None:
            plt.imshow(frame)
            plt.pause(0.001)
        else:
            break
        
    reader.join()
    
def test_save_rtsp():
#    import cv2
#    import os
#    from multiprocessing import Queue
#    from app.framework import QD_Reader, save_and_upload
    with open('config.json','r') as f:
        config=json.load(f)
    video_url=config['video_url']
    reader=QD_Reader(video_url)
    
    idx=0
    filenames=[]
    while True:
        time.sleep(0.1)
        idx=idx+1
        flag,frame=reader.read_from_queue()
        if frame is not None:
            filename=str(time.time())+'.jpg'
            cv2.imwrite(filename,frame)
            filenames.append(filename)
        else:
            print('cannot get image')
            
        if idx>100:
            break
        else:
            print(idx,'save image to video')
    
    assert len(filenames)>0
        
    queue=Queue()
    upload=False
    save_video_name=os.path.join('static','test.mp4')
    try:
        save_and_upload(filenames,save_video_name,queue,upload=upload)
    except Exception as e:
        warnings.warn('exception {}'.format(e.__str__()))
    else:
        if upload:
            fileUrl=queue.get()
            print('upload fileUrl',fileUrl)
        
    reader.join()
    
if __name__ == '__main__':
#     print('test rtsp')
#     test_rtsp()
    print('test rtsp queue')
#    test_rtsp_queue(worker_all)
#    test_rtsp_queue(worker_newest)
#    test_read_from_queue()
    test_save_rtsp()