#!/bin/bash

mesh_obj=$1
case ${mesh_obj} in
	36) scale="1500" ;;
	75) scale="300"	;;
	76) scale="300" ;;
	70) scale="500" ;;
esac
contact_point="contact_point.ply"
texture="surface"
postfix="_example"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ObjectFolder
cd ~/Projects/RandomQuiltingTactile/ObjectFolder

cp ObjectFolder1-100/${mesh_obj}/dataset/${contact_point} \
~/Projects/RandomQuiltingTactile/outputs/${contact_point}
#normal image from ObjectFolder(Taxim)
python model_train.py --mode eval -obj_path ObjectFolder1-100/${mesh_obj} --sample_ply ${contact_point} --object_file_path ObjectFolder1-100/${mesh_obj}/ObjectFile.pth -obj_scale_factor ${scale}
#outputs/tactile_normal_img.png	 <<  normal image
