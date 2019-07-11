#sudo cp qd.service /lib/systemd/system/qd.service
#sudo systemctl daemon-reload
export PORT=8105
export CUDA_VISIBLE_DEVICES=1
#/usr/bin/docker run --runtime nvidia -d -v /home/dell/iscas:/workspace -p ${PORT}:${PORT} youdaoyzbx/pytorch bash -c "cd /workspace/qd && python tasks.py -p ${PORT}"

echo $PORT