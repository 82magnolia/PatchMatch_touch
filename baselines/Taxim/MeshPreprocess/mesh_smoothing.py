import bpy
import math
import argparse
import sys
import os
from os import path as osp

#input_path = "/home/junwon/Datasets/ObjectFolder/1/model.obj"
#output_path = "/home/junwon/Datasets/ObjectFolder/1/smoothed_model.obj"

script_args = sys.argv[sys.argv.index("--") + 1:]

parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", required=True, type=str, help="Path to .obj file")
args = parser.parse_args(script_args)

input_path = args.obj_path
os.makedirs("../result", exist_ok=True)
output_path = osp.join("../result", "smoothed_model.obj")

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

try:
    bpy.ops.wm.obj_import(filepath=input_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=input_path)

print("\nModel Size Analysis and Bevel Application:")

for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        
        # Model size measurement
        dims = obj.dimensions
        max_dim = max(dims)
        print(f"\nModel: {obj.name}")
        print(f"Size: X({dims.x:.3f}), Y({dims.y:.3f}), Z({dims.z:.3f})")
        print(f"Max length: {max_dim:.3f}")

        # Bevel
        obj.modifiers.clear()
        
        bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
        
        bevel.width = max_dim * 0.01            # edge width
        bevel.segments = 8                      # number of face
        bevel.limit_method = 'ANGLE'
        bevel.angle_limit = math.radians(30)    # bevel more than 30 degree 
        
        # Texture protection and anti-overlap
        bevel.miter_outer = 'MITER_ARC'
        bevel.use_clamp_overlap = True 
        
        # Subdivision
        subsurf = obj.modifiers.new(name="Subsurf", type='SUBSURF')
        subsurf.levels = 1 

        bpy.ops.object.modifier_apply(modifier="Bevel")
        bpy.ops.object.modifier_apply(modifier="Subsurf")
        
        bpy.ops.object.shade_smooth()

print("\n" + "="*50)

try:
    bpy.ops.wm.obj_export(filepath=output_path, export_materials=True)
except AttributeError:
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True, use_materials=True)

print(f"Output saved: {output_path}")