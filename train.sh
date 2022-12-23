#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

# settings
GPUS=2,3
EPOCHS=500
BATCH_SIZE=12
IM_SIZE=800
CFG=yolov5m
RESUME=False
SAVE_INTERVAL=100



# dataset
# DATASET="HRSC2016"
# DATASET="UCAS_AOD"
# DATASET="DOTA"
# DATASET="DIOR"
# DATASET="FAIR1M"
# DATASET="IC15"
# DATASET="IC13"
# DATASET="MSRA_TD500"
# DATASET="NWPU_VHR10"
# DATASET="VOC2007"
DATASET="UAV_ROD"

 

###  DP
# python tools/train.py   \
#     --cfg            config/$DATASET/$CFG.yaml    \
#     --data           config/$DATASET/$DATASET.yaml    \
#     --hyp            config/$DATASET/hyp.yaml    \
#     --epochs         $EPOCHS    \
#     --batch-size     $BATCH_SIZE    \
#     --img-size       $IM_SIZE    \
#     --resume         $RESUME    \
#     --name           $DATASET    \
#     --device         $GPUS    \
#     --logdir         runs/    \
#     --save-interval  $SAVE_INTERVAL \
#     --angle_encoding 'BCL8'\
#     --adam     \
#     --noautoanchor   
    

# #####  DDP
python -m torch.distributed.launch --nproc_per_node 2 tools/train.py     \
    --cfg        config/$DATASET/$CFG.yaml    \
    --data       config/$DATASET/$DATASET.yaml    \
    --hyp        config/$DATASET/hyp.yaml    \
    --epochs     $EPOCHS    \
    --batch-size $BATCH_SIZE    \
    --img-size   $IM_SIZE    \
    --resume     $RESUME    \
    --name       $DATASET    \
    --device     $GPUS    \
    --save-interval $SAVE_INTERVAL \
    --angle_encoding 'POE8'\
    --logdir     runs/    \
    --adam     --sync-bn    --noautoanchor


###### ops
    # --adam    \
    # --sync-bn    \
    # --multi-scale    \ 
    # --single-cls    \
    # --image-weights    \
    # --cache-images    \
    # --nosave    \
    # --notest    \
    # --evolve    \
    # --multi-scale     
       
