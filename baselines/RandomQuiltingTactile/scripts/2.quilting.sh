#!/bin/bash

mesh_obj=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ObjectFolder

#image quilting
cd ~/Projects/RandomQuiltingTactile
python ImageQuilting/code/main.py --synthesis -i \
"outputs/tactile_normal_img.png" -s 512 -b 30

#outputs/normal_quilt_img.png
