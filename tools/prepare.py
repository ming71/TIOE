import sys
sys.path.append('.')

import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import tifffile as tif
from bs4 import BeautifulSoup as bs

from utils.general import xywha2xy4
from DOTA_devkit import dota_utils as util 
from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval 
from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_test 

DATASET_LIST = ['HRSC2016', 'UCAS_AOD', 'DOTA', 'UAV_ROD',
                'DIOR', 'FAIR1M',
                'IC15', 'MSRA_TD500',  
                'IC13', 'NWPU_VHR10', 'VOC' ]

def make_yolo_dirs(root_dir, imgsets=['train', 'test']):
    for imset in imgsets:
        set_dir = os.path.join(root_dir, 'yolo_' + imset)
        if os.path.exists(set_dir):
            shutil.rmtree(set_dir)
        os.mkdir(set_dir)
        os.mkdir(os.path.join(set_dir, 'images'))
        os.mkdir(os.path.join(set_dir, 'labels'))       # yolo format
        os.mkdir(os.path.join(set_dir, 'labelTxt'))     # dota format


def make_dirs(set_dir):
    if os.path.exists(set_dir):
        shutil.rmtree(set_dir)
    os.mkdir(set_dir)


def copy_files(src, dst):
    if isinstance(src, str):
        os.system(f'cp -r {src} {dst}')
    elif isinstance(src, list):
        for s, d in zip(src, dst):
            os.system(f'cp  {s} {d}')
    else:
        raise NotImplementedError('what\'s wrong with you, bra?')

def mv_files(src, dst_dir, ext='*.*'):
    for i in glob.glob(os.path.join(src, ext)):
        shutil.move(i, f'{i.replace(src, dst_dir)}')


def tif2png(src, dst_dir):
    pbar = tqdm(glob.glob(os.path.join(src, '*.*')))
    for i in pbar:
        print(i)
        name = os.path.split(i)[1]
        pbar.set_description(f'tif2png in {name}')
        im = tif.imread(i)
        png = i.replace('.tif', '.png')
        tif.imsave(f'{png.replace(src, dst_dir)}', im)

## trans dota format to  (cls, c_x, c_y, Longest side, short side, angle:[0,179))
def dota2LongSideFormat(imgpath, txtpath, dstpath, extractclassname, ext='.png'):
    """
    trans dota farmat to longside format
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  
    os.makedirs(dstpath)  
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_dota_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + ext)  # img_fullname='/.../P000?.jpg'
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            num_gt = 0
            for i, obj in enumerate(objects):

                num_gt = num_gt + 1   
                poly = obj['poly']  
                poly = np.float32(np.array(poly))

                rect = cv2.minAreaRect(poly)  # （(cx,cy), (w,h), a）
                # box = np.float32(cv2.boxPoints(rect))  
                # rect = cv2.minAreaRect(poly)  
                # box = np.float32(cv2.boxPoints(rect))  

                c_x = rect[0][0] /img_w
                c_y = rect[0][1] /img_h
                w = rect[1][0] 
                h = rect[1][1] 
                theta = rect[-1]  # Range for angle is [-90，0)

                trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                if not trans_data:
                    if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
                        print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    num_gt = num_gt - 1
                    continue
                else:
                    # range:[-180，0)
                    c_x, c_y, longside, shortside, theta_longside = trans_data

                bbox = np.array((c_x, c_y, longside/img_w, shortside/img_h))

                if (sum(bbox <= 0) + sum(bbox[:2] >= 1) ) >= 1:  # 0<xy<1, 0<side<=1
                    print('negative bbox in:%s' % (img_fullname))
                    print('longside format:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (c_x, c_y, longside, shortside, theta_longside))
                    num_gt = num_gt - 1
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
                else:
                    print('Not expected class:%s in :%s' % (obj['name'], fullname))
                    num_gt = num_gt - 1
                    continue

                theta_label = theta_longside + 180.0  # range int[0,180] 四舍五入
                if theta_label >= 180:  # range int[0,179]
                    import ipdb;ipdb.set_trace()
                # outline='id x y longside shortside Θ'

                # final check
                if id > len(extractclassname) or id < 0:
                    print('id problems in:%s' % (img_fullname))
                    print('longside format:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                    c_x, c_y, longside, shortside, theta_longside))
                if theta_label < 0 or theta_label >= 180:
                    print('id problems in:%s' % (img_fullname))
                    print('longside format:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox))) + ' ' + str(theta_label)
                f_out.write(outline + '\n')  

        if num_gt == 0:
            os.remove(os.path.join(dstpath, name + '.txt'))  #
            os.remove(img_fullname)
            os.remove(fullname)
            # print('%s 图片对应的txt不存在有效目标,已删除对应图片与txt' % img_fullname)


def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度         
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return: 
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

class DataPrepare(object):
    def __init__(self, dataset, root_dir):
        self.dataset = dataset
        self.root_dir = root_dir
        assert self.dataset in DATASET_LIST, f'Not support {self.dataset}!'

    def prepare(self):
        process = getattr(self, self.dataset.lower())
        process()

    def hrsc2016(self):
        NAMES = ['ship']
        imgsets=['train', 'val', 'trainval', 'test']

        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            lines = open(os.path.join(self.root_dir, 'ImageSets', imgset + '.txt'), 'r').readlines()
            pbar = tqdm(lines)
            for line in pbar:
                # move images
                pbar.set_description(f'processing {line.strip()}.jpg')
                src_im = os.path.join(self.root_dir, 'FullDataSet/AllImages', line.strip() + '.jpg')
                dst_im = os.path.join(self.root_dir, f'yolo_{imgset}/images', line.strip() + '.jpg')
                copy_files(src_im, dst_im)
                # convert xml labels into DOTA format
                src_anno = os.path.join(self.root_dir, 'FullDataSet/Annotations', line.strip() + '.xml')
                dst_anno = os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt', line.strip() + '.txt')
                objs = []
                for obj in bs(open(src_anno), "html.parser").findAll('hrsc_object'):
                    xywha = []
                    xywha.append(float(obj.select_one('mbox_cx').text))
                    xywha.append(float(obj.select_one('mbox_cy').text))
                    xywha.append(float(obj.select_one('mbox_w').text))
                    xywha.append(float(obj.select_one('mbox_h').text))
                    xywha.append(np.rad2deg(float(obj.select_one('mbox_ang').text)))
                    qbb = xywha2xy4(xywha).reshape(-1,8).squeeze().tolist()
                    objs.append(qbb)
                objs = np.array(objs)
                with open(dst_anno, 'w') as f:
                    for k, obj in enumerate(objs):
                        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} ship 0\n'.format(
                            obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7])
                        )

            print(f'Convert to yolo style...')
            dota2LongSideFormat(os.path.join(self.root_dir, f'yolo_{imgset}/images'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labels'),
                                NAMES, ext='.jpg')

    def dior(self):
        NAMES = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'Expressway-Service-area', 'Expressway-toll-station',
             'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
             'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
             'windmill']
        imgsets = ['trainval', 'test']

        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            lines = open(os.path.join(self.root_dir, 'ImageSets/Main', imgset + '.txt'), 'r').readlines()
            pbar = tqdm(lines)
            for line in pbar:
                # move images
                pbar.set_description(f'processing {line.strip()}.jpg')
                src_im = os.path.join(self.root_dir, f'JPEGImages-{imgset}', line.strip() + '.jpg')
                dst_im = os.path.join(self.root_dir, f'yolo_{imgset}/images', line.strip() + '.jpg')
                copy_files(src_im, dst_im)
                # convert xml labels into DOTA format
                src_anno = os.path.join(self.root_dir, 'Annotations/Oriented Bounding Boxes', line.strip() + '.xml')
                dst_anno = os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt', line.strip() + '.txt')
                objs = []
                for obj in bs(open(src_anno, encoding='utf-8'), "html.parser").findAll('object'):
                    qbb = [
                        float(obj.select_one('x_left_top').text),
                        float(obj.select_one('y_left_top').text),
                        float(obj.select_one('x_right_top').text),
                        float(obj.select_one('y_right_top').text),
                        float(obj.select_one('x_right_bottom').text),
                        float(obj.select_one('y_right_bottom').text),
                        float(obj.select_one('x_left_bottom').text),
                        float(obj.select_one('y_left_bottom').text),
                    ]
                    clsname = obj.select_one('name').text
                    objs.append(qbb)
                objs = np.array(objs)
                with open(dst_anno, 'w', encoding='utf-8') as f:
                    for k, obj in enumerate(objs):
                        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {} 0\n'.format(
                            obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], clsname)
                        )

            print(f'Convert to yolo style...')
            dota2LongSideFormat(os.path.join(self.root_dir, f'yolo_{imgset}/images'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labels'),
                                NAMES, ext='.jpg')

    def ucas_aod(self):
        NAMES = ['car', 'airplane']
        imgsets=['train', 'val', 'test']

        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            lines = open(os.path.join(self.root_dir, 'ImageSets', imgset + '.txt'), 'r').readlines()
            pbar = tqdm(lines)
            for line in pbar:
                # move images
                pbar.set_description(f'processing {line.strip()}.png')
                src_im = os.path.join(self.root_dir, 'AllImages', line.strip() + '.png')
                dst_im = os.path.join(self.root_dir, f'yolo_{imgset}/images', line.strip() + '.png')
                copy_files(src_im, dst_im)
                # convert xml labels into DOTA format
                src_anno = os.path.join(self.root_dir, 'Annotations', line.strip() + '.txt')
                dst_anno = os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt', line.strip() + '.txt')
                objs = []
                for obj in open(src_anno, "r").readlines():
                    name, *qbox =  obj.split()[:9]
                    objs.append(' '.join(qbox + [name, '0', '\n'] ))

                with open(dst_anno, 'w') as f:
                        f.write(''.join(objs))

            print(f'Convert to yolo style...')
            dota2LongSideFormat(os.path.join(self.root_dir, f'yolo_{imgset}/images'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labels'),
                                NAMES, ext='.png')

    def ic15(self):
        NAMES = ['text']
        imgsets=['train', 'test']

        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            im_dir = 'ch4_training_images' if imgset=='train' else 'ch4_test_images' 
            anno_dir = 'ch4_training_localization_transcription_gt' if imgset=='train' else 'Challenge4_Test_Task1_GT' 

            # move images
            copy_files(os.path.join(self.root_dir, im_dir) + r'/*.jpg',
                        os.path.join(self.root_dir, f'yolo_{imgset}/images') 
                        )
            pbar = tqdm(glob.glob(os.path.join(self.root_dir, anno_dir) + '/*.txt'))
            for src_anno in pbar:
                filename = os.path.basename(src_anno)
                pbar.set_description(f'processing {filename}')
                # convert txt labels into DOTA format
                dst_anno = os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt', filename.strip('gt_'))
                objs = []
                for obj in open(src_anno, 'r', encoding='utf-8-sig').readlines():
                    qbox =  obj.split(',')[:8]
                    objs.append(' '.join(qbox + [NAMES[0], '0', '\n'] ))

                with open(dst_anno, 'w') as f:
                        f.write(''.join(objs))

            print(f'Convert to yolo style...')
            dota2LongSideFormat(os.path.join(self.root_dir, f'yolo_{imgset}/images'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labels'),
                                NAMES, ext='.jpg')

    def dota(self):         
        NAMES = [ 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
        self.dota_prepare(NAMES)

    def dota_prepare(self, NAMES):
        imgsets=['train', 'val', 'test']
        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            print(f'Split {imgset} set...')
            test_flag = (imgset not in ['train', 'val'])
            if test_flag:
                os.makedirs(os.path.join(self.root_dir, 'test-split', 'images'))
                raw_dir = os.path.join(self.root_dir, imgset, 'images')
                split_dir = os.path.join(self.root_dir, f'{imgset}-split', 'images')
            else:
                raw_dir = os.path.join(self.root_dir, imgset)
                split_dir = os.path.join(self.root_dir, f'{imgset}-split')
            splitbase = splitbase_test  if  test_flag else splitbase_trainval
            split = splitbase(raw_dir,
                      split_dir,
                      gap=200,        
                      subsize=1024,   
                      num_process=32)
            # resize rate before cut
            split.splitdata(0.5)  
            split.splitdata(1)  
            split.splitdata(1.5)  
            # move files
            yolo_img  = os.path.join(self.root_dir, 'yolo_' + imgset + '/images')
            yolo_anno = os.path.join(self.root_dir, 'yolo_' + imgset + '/labelTxt')
            if not test_flag:
                split_imgs = os.path.join(split_dir, 'images')
                split_annos= os.path.join(split_dir, 'labelTxt')
                mv_files(split_annos, yolo_anno, '*.txt')
            else:
                split_imgs = split_dir 
            mv_files(split_imgs, yolo_img, '*.png')
            # convert to yolo annos
            print(f'Convert to yolo style...')
            if not test_flag:
                dota2LongSideFormat(f'{self.root_dir}/yolo_{imgset}/images',
                                    f'{self.root_dir}/yolo_{imgset}/labelTxt',
                                    f'{self.root_dir}/yolo_{imgset}/labels',
                                    NAMES, ext='.png')
            os.system(f'rm -rf {self.root_dir}/{imgset}-split') 
        # Generate trainval set
        print(f'Generate trainval set...')
        make_yolo_dirs(self.root_dir, ['trainval'])
        mv_files(f'{self.root_dir}/yolo_train/images', f'{self.root_dir}/yolo_trainval/images')
        mv_files(f'{self.root_dir}/yolo_train/labels', f'{self.root_dir}/yolo_trainval/labels')
        mv_files(f'{self.root_dir}/yolo_train/labeltxt', f'{self.root_dir}/yolo_trainval/labeltxt')
        mv_files(f'{self.root_dir}/yolo_val/images', f'{self.root_dir}/yolo_trainval/images')
        mv_files(f'{self.root_dir}/yolo_val/labels', f'{self.root_dir}/yolo_trainval/labels')
        mv_files(f'{self.root_dir}/yolo_val/labeltxt', f'{self.root_dir}/yolo_trainval/labeltxt')
        os.system(f'rm -rf {self.root_dir}/yolo_train') 
        os.system(f'rm -rf {self.root_dir}/yolo_val') 




    def fair1m(self):         
        NAMES = [ 'C919', 'ARJ21', 'Tractor', 'Roundabout', 'Trailer',
                'Passenger Ship', 'Warship', 'Football Field', 
                'Truck Tractor', 'Excavator', 'Bus', 'Bridge', 
                'Baseball Field', 'A350', 'Basketball Court', 
                'Engineering Ship', 'Boeing777', 'Tugboat', 'A330', 
                'Boeing787', 'Boeing747', 'other-ship', 'A321', 
                'Tennis Court', 'Liquid Cargo Ship', 'other-vehicle', 
                'Boeing737', 'Fishing Boat', 'A220', 'Intersection', 'Motorboat', 'Cargo Truck', 'Dry Cargo Ship', 
                'other-airplane', 'Dump Truck', 'Van', 'Small Car']

        ## convert xml to txt
        # make_dirs(os.path.join(self.root_dir, 'train', 'labelTxt'))
        # xmls = os.listdir(os.path.join(self.root_dir, 'train', 'labelXml'))
        # pbar = tqdm(xmls)
        # for xml_name in pbar:
        #     pbar.set_description(f'Convert {xml_name} into DOTA txt...')
        #     xml_path = os.path.join(self.root_dir, 'train', 'labelXml', xml_name)
        #     objs = []
        #     for obj in bs(open(xml_path), "html.parser").findAll('object'):
        #         qbb = []
        #         for pts in obj.find('points').findAll('point')[:-1]:
        #             qbb.extend([float(x) for x in pts.text.split(',')])
        #         clsname = obj.select_one('name').text
        #         objs.append(qbb)
        #     objs = np.array(objs)
        #     labelTxt = os.path.join(self.root_dir, 'train', f'labelTxt', xml_name.replace('xml', 'txt'))

        #     with open(labelTxt, 'w', encoding='utf-8') as f:
        #         for k, obj in enumerate(objs):
        #             f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {} 0\n'.format(
        #                 obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], clsname)
        #             )

        ## convert tif to png 
        tif_tr_imgs = os.path.join(self.root_dir, 'train', 'images-tif')
        tif_ts_imgs = os.path.join(self.root_dir, 'test', 'images-tif')
        png_tr_imgs = os.path.join(self.root_dir, 'train', 'images')
        png_ts_imgs = os.path.join(self.root_dir, 'test', 'images')
        # os.system(f'mv {png_tr_imgs} {tif_tr_imgs}') 
        # os.system(f'mv {png_ts_imgs} {tif_ts_imgs}') 
        # make_dirs(png_tr_imgs)
        # make_dirs(png_ts_imgs)
        # tif2png(tif_tr_imgs, png_tr_imgs)
        # tif2png(tif_ts_imgs, png_ts_imgs)

        ## data
        imgsets=['train', 'test']
        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            print(f'Split {imgset} set...')
            test_flag = (imgset not in ['train'])
            if test_flag:
                os.makedirs(os.path.join(self.root_dir, 'test-split', 'images'))
                raw_dir = os.path.join(self.root_dir, imgset, 'images')
                split_dir = os.path.join(self.root_dir, f'{imgset}-split', 'images')
            else:
                raw_dir = os.path.join(self.root_dir, imgset)
                split_dir = os.path.join(self.root_dir, f'{imgset}-split')
            splitbase = splitbase_test  if  test_flag else splitbase_trainval
            split = splitbase(raw_dir,
                      split_dir,
                      gap=200,        
                      subsize=1024,   
                      ext='.png',
                      num_process=32)
            # resize rate before cut
            # split.splitdata(0.5)  
            split.splitdata(1)  
            # split.splitdata(1.5)  

            # move files
            yolo_img  = os.path.join(self.root_dir, 'yolo_' + imgset + '/images')
            yolo_anno = os.path.join(self.root_dir, 'yolo_' + imgset + '/labelTxt')
            if not test_flag:
                split_imgs = os.path.join(split_dir, 'images')
                split_annos= os.path.join(split_dir, 'labelTxt')
                mv_files(split_annos, yolo_anno, '*.txt')
            else:
                split_imgs = split_dir 
            mv_files(split_imgs, yolo_img, '*.png')
            # convert to yolo annos
            print(f'Convert to yolo style...')
            if not test_flag:
                dota2LongSideFormat(f'{self.root_dir}/yolo_{imgset}/images',
                                    f'{self.root_dir}/yolo_{imgset}/labelTxt',
                                    f'{self.root_dir}/yolo_{imgset}/labels',
                                    NAMES, ext='.png')
            os.system(f'rm -rf {self.root_dir}/{imgset}-split') 


    def uav_rod(self):
        NAMES = ['car']
        imgsets=['train', 'test']
        make_yolo_dirs(self.root_dir, imgsets=imgsets)
        for imgset in imgsets:
            print(f'Generate {imgset} with DOTA style...')
            lines = open(os.path.join(self.root_dir, imgset + '.txt'), 'r').readlines()
            pbar = tqdm(lines)
            for line in pbar:
                # move images
                pbar.set_description(f'processing {line.strip()}.jpg')
                src_im = os.path.join(self.root_dir, imgset, 'images', line.strip() + '.jpg')
                dst_im = os.path.join(self.root_dir, f'yolo_{imgset}/images', line.strip() + '.jpg')
                copy_files(src_im, dst_im)
                # convert xml labels into DOTA format
                src_anno = os.path.join(self.root_dir, imgset, 'annotations', line.strip() + '.xml')
                dst_anno = os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt', line.strip() + '.txt')
                objs = []
                for obj in bs(open(src_anno), "html.parser").findAll('robndbox'):
                    xywha = []
                    xywha.append(float(obj.select_one('cx').text))
                    xywha.append(float(obj.select_one('cy').text))
                    xywha.append(float(obj.select_one('w').text))
                    xywha.append(float(obj.select_one('h').text))
                    xywha.append(np.rad2deg(float(obj.select_one('angle').text)))
                    qbb = xywha2xy4(xywha).reshape(-1,8).squeeze().tolist()
                    objs.append(qbb)
                objs = np.array(objs)
                with open(dst_anno, 'w') as f:
                    for k, obj in enumerate(objs):
                        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} car 0\n'.format(
                            obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7])
                        )
            print(f'Convert to yolo style...')
            dota2LongSideFormat(os.path.join(self.root_dir, f'yolo_{imgset}/images'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labelTxt'),
                                os.path.join(self.root_dir, f'yolo_{imgset}/labels'),
                                NAMES, ext='.jpg')



 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DOTA', help='type of dataset')   
    parser.add_argument('--root_dir', type=str, default='data', help='output folder')  
    opt = parser.parse_args()

    print(opt)

    DataPrepare(opt.dataset, opt.root_dir).prepare()