#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

# settings
GPUS=0,1
IM_SIZE=800
CONF_THRES=0.05
IOU_THRES=0.5
USE_VOC07=True
DATASET='UCAS_AOD'
CKPT=runs/exp5_$DATASET/weights/epoch_200.pt



python tools/detect.py   \
    --weights     $CKPT    \
    --output      outputs    \
    --img-size    $IM_SIZE    \
    --conf-thres  $CONF_THRES    \
    --device      $GPUS    \
    --dataset      $DATASET    \
    --source      data/$DATASET/yolo_test/images    \
    --angle_encoding 'POE8'\
    --use-voc07   $USE_VOC07   \
    --save-txt    
    
###### ops
    # --view-img    \
    # --classes    \
    # --agnostic-nms    \
    # --augment    

python tools/evaluation.py  \
    --dataset     $DATASET    \
    --imageset    data/$DATASET/yolo_test/images    \
    --annopath    data/$DATASET/yolo_test/labelTxt/{:s}.txt    \
    --use-voc07   $USE_VOC07     \
    --ovthresh    $IOU_THRES    


# python tools/mAOE_evaluation.py  \
#     --dataset     $DATASET    \


