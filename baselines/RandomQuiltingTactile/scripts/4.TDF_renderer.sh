#!/bin/bash

mesh_obj=$1
contact_point="contact_point.ply"
case ${mesh_obj} in
	36) scale="0.01" ;;
	75) scale="0.05" ;;
	76) scale="0.05" ;;
	70) scale="0.03" ;;
esac
texture="surface"
postfix="_example"

source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate TDF
cd ~/Projects/RandomQuiltingTactile/TactileDreamFusion
sleep 1

echo "Running in testing mode..."
vis_mode="viewspace_normal"
echo "Render: Texture: $texture, Mode: $vis_mode"

# Render contact views as images
python vis_render.py \
    logs/${mesh_obj}_${texture}${postfix}/${mesh_obj}_${texture}${postfix}.obj \
    --mode $vis_mode \
    --elevation 0 \
    --num_azimuth 1 \
    --view_path ../outputs/${contact_point} \
    --scale ${scale} \
    --save ../outputs/

python tactile_simul.py --obj_number ${mesh_obj}
