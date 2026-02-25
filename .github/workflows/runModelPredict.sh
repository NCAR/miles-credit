#! /bin/bash
export PATH=/home/ubuntu/.local/bin:$PATH
#git -C /workspace/miles-credit stash
#git -C /workspace/miles-credit pull -q
echo "miles-credit commit:"
#git -C /workspace/miles-credit log -1
echo "gfs_init.py!"
#conda run -n credit python -u /workspace/miles-credit/applications/gfs_init.py -c /workspace/CIRRUS-MILES-CREDIT/model_predict_old.yml
conda run -n credit python -u /_w/miles-credit/applications/gfs_init.py -c /workspace/CIRRUS-MILES-CREDIT/model_predict_old.yml
echo "rollout_realtime.py!"
#conda run -n credit python -u /workspace/miles-credit/applications/rollout_realtime.py -c ./model_predict_old.yml
conda run -n credit python -u /__w/miles-credit/applications/rollout_realtime.py -c ./model_predict_old.yml
ls -lrth /output

#nvidia-smi
#export PATH=/home/ubuntu/.local/bin:$PATH
#git -C /workspace/miles-credit stash
#git -C /workspace/miles-credit pull -q
#echo "miles-credit commit:"
#git -C /workspace/miles-credit log -1
#echo "gfs_init.py!"
#conda run -n credit python -u /workspace/miles-credit/applications/gfs_init.py -c /workspace/CIRRUS-MILES-CREDIT/model_predict_old.yml
##mkdir -p /output/model_predict
#echo "rollout_realtime.py!"
#conda run -n credit python -u /workspace/miles-credit/applications/rollout_realtime.py -c ./model_predict_old.yml
#ls -lrth /output
