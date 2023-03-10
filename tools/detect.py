import sys
sys.path.append('.')

import os
import cv2
import time
import glob
import shutil
import argparse
import platform
from pathlib import Path
from numpy import random
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, apply_classifier, get_colors, scale_labels,
    xyxy2xywh, plot_one_rotated_box, strip_optimizer, set_logging, rotate_non_max_suppression)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.evaluation_utils import rbox2txt


def detect():
    out, source, weights, view_img, put_text, save_txt, save_img, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.put_text, opt.save_txt, opt.save_img, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)

    if os.path.exists(out):
        shutil.rmtree(out)  
    os.makedirs(out)  


    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = get_colors()

    # Run inference
    t0 = time.time()
    # ????????????????????????,????????????????????????  ???????????????1???3???imgsz???imgsz???
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    """
        path ??????/????????????  'E:\...\bus.jpg'
        img ??????resize+pad???????????????   1*3*re_size1*resize2????????? (3,img_height,img_weight)
        img0 ???size??????   (img_height,img_weight,3)          
        cap ?????????????????????None??????????????????????????????   
    """
    pbar = tqdm(dataset)
    for path, img, im0s, vid_cap in pbar:
        pbar.set_description(f'{path}')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # ??????batch_size????????????????????????????????????
        if img.ndimension() == 3:
            # (in_channels,size1,size2) to (1,in_channels,img_height,img_weight)
            img = img.unsqueeze(0)  # ???[0]?????????????????????

        # Inference
        t1 = time_synchronized()
        """
        model:
        input: in_tensor (batch_size, 3, img_height, img_weight)
        output: ??????????????? [z,x]
        z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), nc)
        x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3???scale???, size1, size2, [xywh,score,num_classes]) 
        '''
               
        ???????????? ??????pred[0]???shape???(1, num_boxes, nc)
        h,w??????????????????????????????????????????dataset????????????????????????????????????????????????h???????????????w
        num_boxes = 3 * h/32 * w/32 + 3 * h/16 * w/16 + 3 * h/8 * w/8
        pred[0][..., 0:4] ??????????????????xywh(?????????+??????)??????
        pred[0][..., 4]???objectness?????????
        pred[0][..., 5:5+nc]???????????????
        pred[0][..., 5+nc:]?????????????????
        """
        # pred : (batch_size, num_boxes, no)  batch_size=1
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # pred : list[tensor(batch_size, num_conf_nms, [xyls??,conf,classid])] ?????[0,179]
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, angle_encoding=opt.angle_encoding)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xyls??,conf,classid]) ?????[0,179]
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)  # ??????????????????+????????????
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            #print(txt_path)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()
                # Print results    det:(num_nms_boxes, [xyls??,conf,classid]) ?????[0,179]
                for c in det[:, -1].unique():  # unique??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    n = (det[:, -1] == c).sum()  # detections per class  ????????????????????????????????????
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string ??????????????? ??????,???

                # Write results  det:(num_nms_boxes, [xywh??,conf,classid]) ?????[0,179]
                for *rbox, conf, cls in reversed(det):  # ??????list???????????????,?????????????????????????????????
                    # rbox=[tensor(x),tensor(y),tensor(w),tensor(h),tsneor(??)] ?????[0,179]
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    label = '%s %.2f' % (names[int(cls)], conf)
                    classname = '%s' % names[int(cls)]
                    conf_str = '%.3f' % conf
                    if save_img or view_img:  # Add bbox to image
                        label = label if put_text else None
                        plot_one_rotated_box(rbox, im0, label=label, color=[0,0,255], line_thickness=1,pi_format=False)
                        # color=[0,0,255]
                        # color=colors[int(cls)]
                    if save_txt:
                        rbox2txt(rbox, classname, conf_str, Path(p).stem, str(out + '/result_txt/result_before_merge'))

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results ????????????
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                    pass
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('   Results saved to %s' % Path(out))

    print('   All Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    """
        weights:???????????????
        source:??????????????????????????????/???????????????????????????'0'(?????????????????????),????????????rtsp????????????
        save-txt:??????????????????????????????txt???????????????????????????False
        classes:??????????????????????????????????????????0??????0 2 3
        agnostic-nms:??????nms?????????????????????????????????????????????False
        augment:????????????????????????????????????????????????(TTA)??????
        update:?????????True???????????????????????????strip_optimizer???????????????pt??????????????????????????????????????????False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/YOLOv5_DOTA_OBB.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='DOTA_demo_view/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='DOTA_demo_view/detection', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--use-voc07', type=str, default='False', help='VOC07 metric')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--angle_encoding', type=str, default='Reg')
    parser.add_argument('--dataset', type=str, default='HRSC2016', help='DATASET')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--put-text', action='store_true', help='show text in imgs')
    parser.add_argument('--save-img', action='store_true', help='save images to output dir')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # ??????pt??????????????????????????????
                strip_optimizer(opt.weights)

        else:
            if os.path.isdir(opt.weights[0]):
                ensemble = False
                all_weights = glob.glob(opt.weights[0] + '/*.pt')
                sort_weight = sorted(all_weights , key=lambda x: int(os.path.basename(x).strip('epoch_.pt')))
                if not ensemble:
                    for w in sort_weight:
                        opt.weights = [w]
                        detect()
                        use_voc07 = opt.use_voc07
                        print(f'\n\nEvaluation with {w}')   
                        os.system(f'python tools/evaluation.py  \
                                    --dataset     {opt.dataset}    \
                                    --imageset    data/{opt.dataset}/yolo_test/images    \
                                    --annopath    data/{opt.dataset}/yolo_test/labelTxt  \
                                    --use-voc07   {use_voc07}')
                else:
                    opt.weights = sort_weight
                    detect()
            else:
                detect()

