# install
- nvidia-driver
- [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
```
sudo usermod -aG docker your-user
```
- pull docker image

## install yolov3
- cd app/config && wget https://pjreddie.com/media/files/yolov3.weights
- mkdir model && cd model && git clone https://github.com/ultralytics/yolov3

# scripts
```
docker pull youdaoyzbx/pytorch:qd_light
git clone https://git.dev.tencent.com/yzbx/qd.git
curl http://10.50.200.171:8080/mtrp/file/json/upload.jhtml -F "file=@/workspace/test.jpg"
```

### systemd service
- sudo cp qd.service /lib/systemd/system/qd.service

### vpn and ssh
view padlist for detail

### miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

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

### video stream
- https://github.com/miguelgrinberg/flask-video-streaming video streaming with Flask + gunicorn + gevent/eventlet

## reference
- [docker start multi service](https://docs.docker.com/config/containers/multi-service_container/)
- [deep learning + redis + flask + apache](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/)

## todo
- [service-streamer boosting web services by queue samples into mini-batches, sacrifice response time](https://github.com/ShannonAI/service-streamer
