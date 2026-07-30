#!/bin/bash

mesh_obj="36"
contact_point="contact_point.ply"
scale="1500"
texture="surface"
postfix="_example"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ObjectFolder
cd ~/Projects/ObjectFolder

cp ObjectFolder1-100/${mesh_obj}/dataset/${contact_point} ~/Projects/outputs/${contact_point}
#normal image from ObjectFolder(Taxim)
python model_train.py --mode eval -obj_path ObjectFolder1-100/${mesh_obj} --sample_ply ${contact_point} --object_file_path ObjectFolder1-100/${mesh_obj}/ObjectFile.pth -obj_scale_factor ${scale}
#outputs/tactile_normal_img.png	 <<  normal image

#image quilting
cd ~/Projects
python ImageQuilting/code/main.py --synthesis -i "outputs/tactile_normal_img.png" -s 1024
#outputs/normal_quilt_img.png

#TDF train
conda deactivate
conda activate TDF
cd ~/Projects/TactileDreamFusion

cp ~/Projects/outputs/normal_quilt_img.png ./data/tactile_textures/${texture}_tactile_texture_map_2_normal.png


#TextureDreambooth/output/lora_{quilting_image}_sks << copy from other
for ((i=0; i<5; i++)); do
	cp ~/Projects/outputs/tactile_normal_img.png \
	TextureDreambooth/output/lora_${texture}_sks/inference_output_guidance_10_${i}.jpg
done



echo "Running in training mode..."
CUDA_VISIBLE_DEVICES=0 python main.py \
--config configs/text_tactile_TSDS.yaml \
save_path=${mesh_obj}_${texture}${postfix} \
mesh=data/base_meshes/${mesh_obj}/model.obj \
tactile_texture_object=${texture} \

###############################################################


#TDF render
#contact point (x,y,z,nz,ny,nz) need to add
#tactile simul >> contact_point_normal()
#####
echo "Running in testing mode..."
vis_modes=("viewspace_normal")
for ((q=0; q<${#vis_modes[@]}; q++)); do
        vis_mode=${vis_modes[$q]}
        echo "Render: Texture: $texture, Mode: $vis_mode"
	
        # Render contact views as images
        python vis_render.py \
            logs/${mesh_obj}_${texture}${postfix}/${mesh_obj}_${texture}${postfix}.obj \
            --mode $vis_mode \
            --elevation 0 \
            --num_azimuth 1 \
            --view_path ../outputs/${contact_point} \
            --scale 1.3 \
            --save ../outputs/ &
done
# ./logs/${mesh_obj}_${texture}${postfix}/0_0_light_0_0_0.1_viewspace_normal.png


# Normal to Height
python tactile_simul.py --obj_number ${mesh_obj}
# normal/${mesh_obj}_height_map_resize.npy

# Heigth to taxim
conda deactivate
conda activate ObjectFolder
cd ~/Projects/ObjectFolder

python model_train.py --mode eval -obj_path ObjectFolder1-100/${mesh_obj} --object_model ObjectFile.pth --sample_ply ${contact_point} --object_file_path ObjectFolder1-100/${mesh_obj}/ObjectFile.pth --TDF_num ${mesh_obj}



