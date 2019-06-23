# install
- nvidia-driver
- nvidia-docker
- pull docker image

## install yolov3
- cd app/config && wget https://pjreddie.com/media/files/yolov3.weights
- mkdir model && cd model && git clone https://github.com/ultralytics/yolov3

# scripts
```
docker pull youdaoyzbx/pytorch:1.1
git clone https://git.dev.tencent.com/yzbx/qd.git
curl http://10.50.200.171:8080/mtrp/file/json/upload.jhtml -F "file=@/workspace/test.jpg"
```

### vpn and ssh
view padlist for detail

### demand
- save video and image
- alarm by web app
- rtsp video stream process
- web interface
- store information to mysql database
- upload video and image to target server

### database
- sqlalchemy and splite
```
# Unix/Mac - 4 initial slashes in total
engine = create_engine('sqlite:////absolute/path/to/foo.db')
```

### source code orgnization
```
.
├── app # application, code to use and modiry open source model
│   ├── config # config file for weight file, model define and classes name
│   └── ...
├── model # open source model
│   └── yolov3 -> /home/yzbx/git/gnu/code/yolov3 # open source model
└── templates # html

```