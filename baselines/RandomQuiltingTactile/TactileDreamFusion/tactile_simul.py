import numpy as np
import trimesh
import cv2
from os import path
import os
from poisson_solver import poisson_dct_neumann
import argparse

def get_normal_from_uv(obj, normal_map, object_point):
    mesh = trimesh.load(obj, process=False) 
    
    texture = cv2.imread(normal_map, cv2.IMREAD_UNCHANGED)
    #texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    h, w = texture.shape[:2]

    _, _, triangle_id = mesh.nearest.on_surface([object_point])
    index = triangle_id[0]

    # Triangular mesh including points
    face_vertices = mesh.vertices[mesh.faces[index]]
    
    # Triangular weights
    barycentric = trimesh.triangles.points_to_barycentric([face_vertices], [object_point])[0]
    
    # Triangular UV including points (3, 2)
    face_uvs = mesh.visual.uv[mesh.faces[index]]
    uv = (barycentric[:, np.newaxis] * face_uvs).sum(axis=0)
    u, v = uv[0], uv[1]

    # UV (0~1) -> Pixel (0~W, 0~H)
    px = int(u * (w - 1))
    py = int((1 - v) * (h - 1))
    
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    rgb = texture[py, px]                                           # 0 ~ 255
    normal_vector = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0    # -1.0 ~ 1.0

    width = 160 * 4
    height = 120 * 4

    start_x = max(0, px - width//2)
    start_y = max(0, py - height//2)
    end_x = min(w, px + width//2)
    end_y = min(h, py + height//2)

    patch_img = texture[start_y:end_y, start_x:end_x].copy()

    return patch_img, normal_vector, (u, v)


def contact_point_normal(obj, contact_point, contact_theta=0., obj_scale_factor = 1000):
    mesh = trimesh.load(obj, force='mesh', process=True)
    mesh.vertices = np.asarray(mesh.vertices) * obj_scale_factor
    mesh.fix_normals()
    proximitry_query = trimesh.proximity.ProximityQuery(mesh)
    vertex_normals = mesh.vertex_normals
    
    sim_vertices = np.copy(mesh.vertices)

    if contact_point is not None:
        cx = contact_point[0] * obj_scale_factor
        cy = contact_point[1] * obj_scale_factor
        cz = contact_point[2] * obj_scale_factor

        # Contact point array
        contact_arr = np.array([cx, cy, cz])

        # Find normal at contact point
        nn_point, _, nn_fid = proximitry_query.on_surface(contact_arr.reshape(1, 3))
        nn_bary = trimesh.triangles.points_to_barycentric(mesh.triangles[nn_fid], points=contact_arr.reshape(1, 3))
        new_normal = trimesh.unitize((mesh.vertex_normals[mesh.faces[nn_fid]] * trimesh.unitize(nn_bary).reshape(-1, 3, 1)).sum(axis=1))
        new_normal = new_normal.reshape(-1)
    
    return new_normal

def get_depth_from_normal(normal_map_path, mask_threshold=None):
    # image size : 800,800
    image = cv2.imread(normal_map_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image[160:640, 80:720]
    bg_mask = np.all(image > 250, axis=2)

    normal_map = (image.astype(np.float32) / 255.0) * 2.0 - 1.0
    normal_x = normal_map[:, :, 0]
    normal_y = normal_map[:, :, 1]
    normal_z = normal_map[:, :, 2]

    epsilon = 1e-6
    normal_z[np.abs(normal_z) < epsilon] = epsilon
    
    gradient_x = -normal_x / normal_z
    gradient_y = -normal_y / normal_z

    gradient_x[bg_mask] = 0
    gradient_y[bg_mask] = 0

    # Poisson Integration
    depth_map = poisson_dct_neumann(gx=gradient_x, gy=gradient_y)
    h, w = image.shape[:2]
    depth_map = depth_map.reshape((h, w))
    if np.any(~bg_mask):
        obj_height = depth_map[~bg_mask]
        obj_min = obj_height.min()
        obj_max = obj_height.max()
        
        depth_map[~bg_mask] = (depth_map[~bg_mask] - obj_min) / (obj_max - obj_min + epsilon) * 255
        depth_map[bg_mask] = 255

    return depth_map

def depth_normalize(depth_map, threshold=None):
    epsilon = 1e-6
    mask = np.zeros((depth_map.shape[0], depth_map.shape[1]))
    if threshold is not None:
        depth_map[depth_map < threshold] = threshold
        mask[depth_map >= threshold] = 255
        #depth_map[depth_map >= threshold] -= threshold

    if False:
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + epsilon)
        depth_map_norm = np.clip(depth_map_norm, 0, 1) * 255.0
    else:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + epsilon)
        depth_map_norm = depth_map
    
    return mask, depth_map_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_number', type=str, help="ObjectFolder number")
    parser.add_argument('--scale', type=float, help="TDF scale")
    args = parser.parse_args()
    
    obj_number = args.obj_number
    #obj_file = os.path.join('logs', obj_number, "model.obj")
    texture = "sample"

    
    normal_map_path = os.path.join('..', "outputs", 'viewspace_normal.png')
    depth_map = get_depth_from_normal(normal_map_path)
    
    #cv2.imwrite(os.path.join('normal', f'{obj_number}_height_map.png'), depth_map)
    #depth_map = cv2.resize(depth_map, (640,480))
    #mask, depth_map_resize = depth_normalize(depth_map)
    
    cv2.imwrite(os.path.join('..', 'outputs', f'{obj_number}_height_map_resize.png'), depth_map)
    np.save(os.path.join('..', 'outputs', f'{obj_number}_height_map_resize.npy'),depth_map)

