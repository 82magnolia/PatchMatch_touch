#!/bin/bash

mesh_obj=$1
texture="surface"
postfix="_example"

source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate TDF
cd ~/Projects/RandomQuiltingTactile/TactileDreamFusion

cp ~/Projects/RandomQuiltingTactile/outputs/normal_quilt_img.png \
./data/tactile_textures/${texture}_tactile_texture_map_2_normal.png


#TextureDreambooth/output/lora_{quilting_image}_sks << copy from other
for ((i=0; i<5; i++)); do
	cp ~/Projects/RandomQuiltingTactile/outputs/tactile_normal_img.png \
	TextureDreambooth/output/lora_${texture}_sks/inference_output_guidance_10_${i}.jpg
done



echo "Running in training mode..."
CUDA_VISIBLE_DEVICES=0 python main.py \
--config configs/text_tactile_TSDS.yaml \
save_path=${mesh_obj}_${texture}${postfix} \
mesh=data/base_meshes/${mesh_obj}/model.obj \
tactile_texture_object=${texture} \
