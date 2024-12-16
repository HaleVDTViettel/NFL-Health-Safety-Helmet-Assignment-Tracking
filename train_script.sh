cd training/tmp/nfl_extra/yolov5
python train.py --img 720 \
                 --batch 16 \
                 --epochs 10 \
                 --data data/data.yaml \
                 --weights yolov5s.pt \
                 --save-period 1\
                 --project nfl-extra