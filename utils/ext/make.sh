#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

clear


cd  rbbox_overlap_gpu
if [ -d "build" ]; then
    rm -r build *.so
fi
python setup.py build_ext --inplace

cd ../rotate_overlap_diff/cuda_op
if [ -d "build" ]; then
    rm -r build *.so
fi
python setup.py install

cd ../..



