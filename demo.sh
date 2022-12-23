#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

# settings
GPUS=0,
IM_SIZE=800
CONF_THRES=0.5
IOU_THRES=0.1
DATASET='HRSC2016'
CKPT=runs/exp8_$DATASET/weights/epoch_299.pt




python tools/detect.py   \
    --weights     $CKPT    \
    --output      outputs    \
    --img-size    $IM_SIZE    \
    --conf-thres  $CONF_THRES    \
    --iou-thres   $IOU_THRES    \
    --device      $GPUS    \
    --source      data/$DATASET/Test/    \
    --angle_encoding 'POE8'\
    --save-img    


    # ops
    # --view-img    \
    # --save-txt    \
    # --multi-scale    \
    # --classes    \
    # --agnostic-nms    \
    # --put-text       \ 


