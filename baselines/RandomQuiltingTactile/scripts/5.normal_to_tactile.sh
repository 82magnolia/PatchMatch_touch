#!/bin/bash

mesh_obj=$1
contact_point="contact_point.ply"
case ${mesh_obj} in
	36) scale="1500" ;;
	75) scale="300"	;;
	76) scale="300" ;;
	70) scale="500" ;;
esac
texture="surface"
postfix="_example"

source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ObjectFolder
cd ~/Projects/RandomQuiltingTactile/ObjectFolder

python model_train.py --mode eval -obj_path ObjectFolder1-100/${mesh_obj} --object_model ObjectFile.pth --sample_ply ${contact_point} --object_file_path ObjectFolder1-100/${mesh_obj}/ObjectFile.pth --TDF_num ${mesh_obj} -obj_scale_factor ${scale}
