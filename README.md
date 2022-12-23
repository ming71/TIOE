# TIOE-Det

This project hosts the official implementation for the paper: 

**Task Interleaving and Orientation Estimation for High-Precision Oriented Object Detection in Aerial Images** 
<!-- [[paper](https://ieeexplore.ieee.org/abstract/document/9488629)]  -->

<!-- ( accepted by **ISPRS Journal of Photogrammetry and Remote Sensing**).  -->



## Abstract
 In this paper, we propose a  Task Interleaving and Orientation Estimation Detector for high-quality oriented object detection in aerial images. Specifically, a posterior hierarchical alignment (PHA) indicator is proposed to optimize the detection pipeline. TIOE-Det adopts PHA indicator to integrate fine-grained posterior localization guidance into classification task to address the misalignment between classification and localization subtasks. Then, a balanced alignment loss is developed to solve the imbalance localization loss contribution in PHA prediction. Moreover, we propose a progressive orientation estimation (POE) strategy to approximate the orientation of objects with n-ary codes. On this basis, an angular deviation weighting strategy is proposed to achieve accurate evaluation of angle deviation in POE strategy.

## Framework
![framework](docs/model.jpg) 

### Setup
```
conda create -n tioe python=3.6 -y
source activate tioe
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

sudo apt-get install swig
pip install -r requirements.txt

cd  DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
cd ..

sh compile.sh
```



### Training
1. Creat config files.
2. Dataset transformation via running `sh prepare.sh`.
3. Run `sh train.sh`.


### Inference & Testing
Run `sh demo.sh` and Run `sh test.sh`.

## Visualizations
![demo](./docs/demo.jpg) 


<!-- ## Citation

If you find our work or code useful in your research, please consider citing:

```
@article{ming2021cfc,
    author={Ming, Qi and Miao, Lingjuan and Zhou, Zhiqiang and Dong, Yunpeng},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    title={CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote-Sensing Images},
    year={2021},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095186}
}


``` -->

Feel free to contact [me](chaser.ming@gmail.com)  if there are any questions.

