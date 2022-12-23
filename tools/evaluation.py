import sys
sys.path.append('.')

import os
import re
import glob
import codecs
import shutil
import zipfile
import argparse
import numpy as np
from DOTA_devkit import polyiou
from functools import partial
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from utils.evaluation_utils import mergebypoly, evaluation_trans, image2txt, draw_DOTA_image

def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                # if (len(splitlines) == 9):
                #     object_struct['difficult'] = 0
                # elif (len(splitlines) == 10):
                #     object_struct['difficult'] = int(splitlines[9])
                object_struct['difficult'] = 0
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects







############################################################
############################################################
############################################################
############################################################
############################################################

class NAMES(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def get_names(self):
        return getattr(self, self.dataset)()
    
    def HRSC2016(self):
        return ['ship']

    def UCAS_AOD(self):
        return ['car', 'airplane']

    def NWPU_VHR10(self):
        return ['airplane', 'ship', 'storage-tank','baseball-diamond', 
            'tennis-court', 'basketball-court', 'ground-track-field', 
            'harbor', 'bridge', 'vehicle']

    def VOC(self):
        return ['aeroplane','bicycle','bird','boat','bottle','bus',
            'car','cat','chair','cow','diningtable','dog',
            'horse','motorbike','person','pottedplant','sheep',
            'sofa', 'train','tvmonitor']

    def UAV_ROD(self):
        return ['car']

    def DOTA(self):
        return ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

    def DOTA1_5(self):
        return ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    def DOTA2(self):
        return ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad']

    def DIOR(self):
        return ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'Expressway-Service-area', 'Expressway-toll-station',
             'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
             'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
             'windmill']

    def FAIR1M(self):
        return ['C919', 'ARJ21', 'Tractor', 'Roundabout', 'Trailer',
                'Passenger Ship', 'Warship', 'Football Field', 
                'Truck Tractor', 'Excavator', 'Bus', 'Bridge', 
                'Baseball Field', 'A350', 'Basketball Court', 
                'Engineering Ship', 'Boeing777', 'Tugboat', 'A330', 
                'Boeing787', 'Boeing747', 'other-ship', 'A321', 
                'Tennis Court', 'Liquid Cargo Ship', 'other-vehicle', 
                'Boeing737', 'Fishing Boat', 'A220', 'Intersection', 'Motorboat', 'Cargo Truck', 'Dry Cargo Ship', 
                'other-airplane', 'Dump Truck', 'Van', 'Small Car']


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap



def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    # pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            # arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, filename)
    zipf.close()




def dota_evaluation(detoutput):
    result_before_merge_path = str(detoutput + '/result_txt/result_before_merge')
    result_merged_path = str(detoutput + '/result_txt/result_merged')
    result_classname_path = str(detoutput + '/result_txt/result_classname')

    mergebypoly(
        result_before_merge_path,
        result_merged_path
    )
    evaluation_trans(
        result_merged_path,
        result_classname_path
    )
    zip_file = os.path.join(detoutput, 'submit.zip')
    if os.path.exists(zip_file):
        os.remove(zip_file)
    make_zip(result_classname_path, zip_file)
    print('Finished!')



def evaluation(detoutput, imageset, annopath, classnames, use_07_metric=True, ovthresh=0.5):
    """
    @param detoutput: detect.py函数的结果存放输出路径
    @param imageset: # val DOTA原图数据集图像路径
    @param annopath: 'r/.../{:s}.txt'  原始val测试集的DOTAlabels路径
    @param classnames: 测试集中的目标种类
    """
    result_before_merged_path = str(detoutput + '/result_txt/result_before_merge')
    result_classname_path = str(detoutput + '/result_txt/result_classname')
    imageset_name_file_path = str(detoutput + '/result_txt')

    evaluation_trans(
        result_before_merged_path,
        result_classname_path
    )
    # print('检测结果已按照类别分类')
    image2txt(
        imageset,  # val原图数据集路径
        imageset_name_file_path
              )
    # print('校验数据集名称文件已生成')

    detpath = str(result_classname_path + '/Task1_{:s}.txt')  # 'r/.../Task1_{:s}.txt'  存放各类别结果文件txt的路径
    annopath = annopath
    imagesetfile = str(imageset_name_file_path +'/imgnamefile.txt')  # 'r/.../imgnamefile.txt'  测试集图片名称txt

    classaps = []
    map = 0
    skippedClassCount = 0
    for classname in classnames:
        print('classname:', classname)
        detfile = detpath.format(classname)
        if not (os.path.exists(detfile)):
            skippedClassCount += 1
            print('This class is not be detected in your dataset: {:s}'.format(classname))
            continue
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=ovthresh,
             use_07_metric=use_07_metric)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        #  save p-r curve of each category
        plt.figure(figsize=(8,4))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec)
        plt.savefig(os.path.join(detoutput, f'pr_{classname}.png'))
    map = map/(len(classnames)-skippedClassCount)
    print('-------------------')
    print('map:', map)
    classaps = 100*np.array(classaps)
    # print('classaps: ', classaps)


def text_evaluation(out_dir, eval_dir='utils/text_eval'):
    tmp_dir = os.path.join(eval_dir, 'temp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    det_files = os.path.join(out_dir, 'result_txt/result_before_merge/*.txt')
    for det_file in glob.glob(det_files):
        im_name = os.path.basename(det_file)
        dst_file = os.path.join(tmp_dir, f'res_{im_name}')
        with open(det_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        dets = [','.join(line.split()[2:-1]) for line in lines]
        with codecs.open(dst_file, 'w', 'utf-8') as f:
            f.write('\n'.join(dets))

    zip_name = 'submit.zip'
    make_zip(tmp_dir, os.path.join(eval_dir, zip_name))
    result = os.popen('cd {0} && python script.py -g=gt.zip -s=submit.zip '.format(eval_dir)).read()
    p = eval(re.findall('(?<="precision": ).*$', result)[0][:6])*100
    r = eval(re.findall('(?<="recall": ).*$', result)[0][:6])*100
    f1 = eval(re.findall('(?<="hmean": ).*$', result)[0][:6])*100
    for name, metric  in zip(['precision', 'recall', 'hmean'], [p, r, f1]):
        print(f'{name}: {metric}')
    
    shutil.rmtree(tmp_dir)



def test():

    out_dir, im_dir, anno_dir,  dataset, use_voc07 ,ovthresh = opt.detoutput, opt.imageset, opt.annopath,  opt.dataset, opt.use_voc07=='True', opt.ovthresh
    if dataset in ['MSRA_TD500', 'IC15', 'IC13']:
        text_evaluation(out_dir, f'utils/text_eval/{dataset.lower()}')
        
    elif dataset in ['HRSC2016', 'UCAS_AOD', 'UAV_ROD','VOC', 'NWPU_VHR10','DIOR']:
        classnames = NAMES(dataset).get_names()
        evaluation(
            detoutput = 'outputs',
            imageset  = f'data/{dataset}/yolo_test/images',
            annopath  = f'data/{dataset}' + r'/yolo_test/labelTxt/{:s}.txt',
            classnames = classnames, 
            use_07_metric=use_voc07,
            ovthresh=ovthresh)
    elif dataset in ['DOTA', 'DOTA1_5', 'DOTA2']:
        classnames = NAMES(dataset).get_names()
        dota_evaluation('outputs')
    else:
        raise NotImplementedError('Not supported dataset!')
 


if __name__ == '__main__':
    '''
    计算AP的流程:
    1.detect.py检测一个文件夹的所有图片并把检测结果按照图片原始来源存入 原始图片名称.txt中:   (rbox2txt函数)
        txt中的内容格式:  目标所属图片名称_分割id 置信度 poly classname
    2.ResultMerge.py将所有 原始图片名称.txt 进行merge和nms,并将结果存入到另一个文件夹的 原始图片名称.txt中:
        txt中的内容格式:  目标所属图片名称 置信度 poly classname
    3.写一个evaluation_trans.py将上个文件夹中的所有txt中的目标提取出来,按照目标类别分别存入 Task1_类别名.txt中:
        txt中的内容格式:  目标所属原始图片名称 置信度 poly
    4.写一个imgname2txt.py 将测试集的所有图片名称打印到namefile.txt中
    '''
    # draw_DOTA_image(imgsrcpath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_images',
    #                 imglabelspath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged',
    #                 dstpath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/merged_drawed',
    #                 extractclassname=classnames,
    #                 thickness=2
    #                 )


    parser = argparse.ArgumentParser()
    parser.add_argument('--detoutput', type=str, default='outputs', help='outputs dir')  
    parser.add_argument('--imageset', type=str, default='./images', help='images folder')  
    parser.add_argument('--annopath', type=str, default='./labels', help='dota label folder')  
    parser.add_argument('--dataset', type=str, default='DOTA', help='type of dataset')
    parser.add_argument('--use-voc07', type=str, default='False', help='VOC07 metric')
    parser.add_argument('--ovthresh', type=float, default=0.5, help='IoU thers')
    opt = parser.parse_args()

    print(opt)

    test()