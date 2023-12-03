''' 
Generate uv position map of 300W_LP.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import skimage.transform
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def crop_and_shift_mesh(vertices, faces, z_offset):
    """
    Efficiently crop a mesh at a specified Z offset and shift it to z=0.

    :param vertices: Nx3 numpy array of vertices.
    :param faces: Mx3 numpy array of faces (indices into vertices).
    :param z_offset: Z value at which to crop the mesh.
    :return: Cropped and shifted vertices and faces.
    """
    # Filter vertices below the z_offset
    valid_vertices_mask = vertices[:, 2] >= z_offset
    cropped_vertices = vertices[valid_vertices_mask]

    # Shift the mesh so the base lies at z=0
    min_z = np.min(cropped_vertices[:, 2])
    cropped_vertices[:, 2] -= min_z

    # Create a mapping from old vertex indices to new indices
    index_mapping = np.full(vertices.shape[0], -1, dtype=int)
    index_mapping[valid_vertices_mask] = np.arange(cropped_vertices.shape[0])

    # Update faces
    mapped_faces = index_mapping[faces]
    valid_faces_mask = np.all(mapped_faces >= 0, axis=1)
    cropped_faces = mapped_faces[valid_faces_mask]

    return cropped_vertices, cropped_faces

def get_vertex_position_on_radial_map(vertex, face_center, res_phi, res_y, min_y, max_y):
    r, phi, y = cartesian_to_cylindrical_vertical_axis(vertex, face_center)

    leftmost_phi = -np.pi/2
    rightmost_phi = np.pi/2
    if leftmost_phi <= phi <= rightmost_phi:
        phi_normalized = (phi - leftmost_phi) / (rightmost_phi - leftmost_phi)
        y_normalized = (y - min_y) / (max_y - min_y)

        i = int(phi_normalized * (res_phi - 1))
        j = res_y - 1 - int(y_normalized * (res_y - 1))

        return r, i, j
    else:
        return None

def map_position_to_vertex(i, j, r, face_center, res_phi, res_y, min_y, max_y):
    # Denormalize the map coordinates
    phi_normalized = i / (res_phi - 1)
    y_normalized = 1 - j / (res_y - 1)

    # Convert normalized coordinates back to original phi and y
    phi = phi_normalized * (np.pi) - np.pi / 2
    y = y_normalized * (max_y - min_y) + min_y

    # Convert from cylindrical back to Cartesian coordinates
    x = np.cos(phi) * r + face_center[0]
    z = np.sin(phi) * r
    return x, y, z

def cartesian_to_cylindrical_vertical_axis(vertex, face_center):
    x, z, y = vertex[0] - face_center[0], vertex[2], vertex[1]
    r = np.sqrt(x**2 + z**2)
    phi = np.arctan2(x, z)
    return r, phi, y

def find_bounding_box_and_vertices_of_triangle(triangle, face_center, res_phi, res_y, min_y, max_y):
    results = [get_vertex_position_on_radial_map(vertex, face_center, res_phi, res_y, min_y, max_y) for vertex in triangle]
    results = [result for result in results if result is not None]

    if len(results) != 3:
        return None, None, None

    # Unpack results
    depths, indices = zip(*[(result[0], (result[1], result[2])) for result in results])
    indices_i, indices_j = zip(*indices)

    # Calculate bounding box
    min_i, max_i = min(indices_i), max(indices_i)
    min_j, max_j = min(indices_j), max(indices_j)
    bounding_box = (min_i, max_i, min_j, max_j)

    return bounding_box, depths, indices

def update_interpolated_map_normalized(interpolated_map_normalized, bbox, interpolated_map):
    min_i, max_i, min_j, max_j = bbox
    slice_i = slice(min_i, max_i + 1)
    slice_j = slice(min_j, max_j + 1)

    # Update the normalized map
    interpolated_map_normalized[slice_j, slice_i] = np.maximum(
        interpolated_map_normalized[slice_j, slice_i],
        interpolated_map[:max_j - min_j + 1, :max_i - min_i + 1]
    )

def interpolate_depth_within_bounding_box(map_vertices, triangle_depths, bounding_box, res_phi, res_y):
    min_i, max_i, min_j, max_j = bounding_box
    bbox_height = max_j - min_j + 1
    bbox_width = max_i - min_i + 1
    interpolated_depths = np.zeros((bbox_height, bbox_width))

    # Preparing for vectorized barycentric coordinates calculation
    tri_vertices = np.array([[v[0], v[1]] for v in map_vertices])
    A, B, C = tri_vertices
    v0 = B - A
    v1 = C - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01

    # Create a grid of points within the bounding box
    i_grid, j_grid = np.meshgrid(np.arange(min_i, max_i + 1), np.arange(min_j, max_j + 1))
    points = np.vstack([i_grid.ravel(), j_grid.ravel()]).T

    # Calculate barycentric coordinates for all points in the grid
    v2 = points - A
    v0_2d = v0[np.newaxis, :]
    v1_2d = v1[np.newaxis, :]
    d20 = np.einsum('ij,ji->i', v2, v0_2d.T)
    d21 = np.einsum('ij,ji->i', v2, v1_2d.T)

    if np.abs(denom) < 1e-10:
        return np.zeros_like(i_grid, dtype=float)
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # Identify points inside the triangle
    inside_triangle = (u >= 0) & (v >= 0) & (w >= 0)
    valid_points = points[inside_triangle]
    valid_u = u[inside_triangle]
    valid_v = v[inside_triangle]
    valid_w = w[inside_triangle]

    inside_triangle_reshaped = inside_triangle.reshape(bbox_height, bbox_width)

    # Interpolate depth values for points inside the triangle
    interpolated_depths = np.zeros_like(i_grid, dtype=float)
    interpolated_depths[inside_triangle_reshaped] = triangle_depths[0] * valid_u + triangle_depths[1] * valid_v + triangle_depths[2] * valid_w

    return interpolated_depths

def generate_cylindrical_radius_map(vertices, faces, out_h, out_w):
    # crop out ears and shift to z=0
    face_center = np.mean(vertices, axis=0)
    center_x, center_y, center_z = face_center[0], face_center[1], face_center[2]
    y_min_index = np.argmin(vertices[:, 1])
    vertex_at_y_min = vertices[y_min_index]
    z_offset = vertex_at_y_min[2] - center_z
    center_z += z_offset
    cropped_vertices, cropped_faces = crop_and_shift_mesh(vertices, faces, center_z)

    res_phi, res_y = out_w, out_h
    min_y, max_y = np.min(cropped_vertices[:,1]), np.max(cropped_vertices[:,1])

    facy_map_norm = np.zeros((res_y, res_phi))
    for triangle_indices in cropped_faces:
        triangle_vertices = [cropped_vertices[index] for index in triangle_indices]
        bbox, triangle_depths, triangle_map_vertices = find_bounding_box_and_vertices_of_triangle(triangle_vertices, face_center, res_phi, res_y, min_y, max_y)
        if bbox is None:
            continue

        facy_map = interpolate_depth_within_bounding_box(triangle_map_vertices, triangle_depths, bbox, res_phi, res_y)
        update_interpolated_map_normalized(facy_map_norm, bbox, facy_map)
    
    return facy_map_norm

def run_facymap_300W_LP(bfm, image_path, mat_path, save_folder, cr_map = None, out_h = 256, out_w = 256, image_h = 256, image_w = 256):
    # 1. load image and fitted parameters
    image_name = image_path.strip().split('/')[-1]
    image = io.imread(image_path)/255.
    [h, w, c] = image.shape

    info = sio.loadmat(mat_path)
    pose_para = info['Pose_Para'].T.astype(np.float32)
    shape_para = info['Shape_Para'].astype(np.float32)
    exp_para = info['Exp_Para'].astype(np.float32)

    # 2. generate mesh
    # generate shape
    vertices = bfm.generate_vertices(shape_para, exp_para)
    # transform mesh
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    image_vertices[:,1] = h - image_vertices[:,1] - 1

    # 3. crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0, 
             bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top)/2
    size = int(old_size*1.5)
    # random pertube. you can change the numbers
    marg = old_size*0.1
    t_x = np.random.rand()*marg*2 - marg
    t_y = np.random.rand()*marg*2 - marg
    center[0] = center[0]+t_x; center[1] = center[1]+t_y
    size = size*(np.random.rand()*0.2 + 0.9)

    # crop and record the transform parameters
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))

    # transform face position(image vertices) along with 2d facial image 
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2]*tform.params[0, 0] # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2]) # translate z

    if cr_map is None:
        cr_map = generate_cylindrical_radius_map(vertices, bfm.full_triangles, out_h, out_w)

    cropped_image = np.clip(cropped_image, 0, 1)
    cropped_image_8bit = (cropped_image * 255).astype(np.uint8)
    # 5. save files
    io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image_8bit))
    np.savez_compressed('{}/{}'.format(save_folder, image_name.replace('jpg', 'npz')), cr_map)
    cr_map_norm = (cr_map)/np.max(cr_map)
    cr_map_8bit = (cr_map_norm * 255).astype(np.uint8)
    io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_facymap.jpg')), cr_map_8bit)
    return cr_map

    # --verify
    # import cv2
    # uv_texture_map_rec = cv2.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_tex.jpg')), np.squeeze(uv_texture_map_rec))

def get_base_image_ids(subdir_path):
    """
    Extract base part of the image filenames, excluding the final augmentation index.

    :param subdir_path: Path to the directory containing images.
    :return: A sorted list of unique base image identifiers.
    """
    all_files = os.listdir(subdir_path)
    base_image_ids = set()

    for file in all_files:
        if file.endswith('.jpg'):
            # Remove the last part (augmentation index) and join back
            base_id = '_'.join(file.split('_')[:-1])
            base_image_ids.add(base_id)

    return sorted(base_image_ids)

def get_augmented_images(base_image_id, subdir_path):
    """
    Get a sorted list of all images that match the base image ID.

    :param base_image_id: The base image ID.
    :param subdir_path: Path to the directory containing images.
    :return: A sorted list of filenames that start with the base image ID.
    """
    augmented_images = []
    all_files = os.listdir(subdir_path)

    for file in all_files:
        if file.startswith(base_image_id) and file.endswith('.jpg'):
            augmented_images.append(file)

    augmented_images.sort()
    return augmented_images

def process_dataset(bfm, dataset_dir, save_folder, uv_coords, uv_h=256, uv_w=256, image_h=256, image_w=256):
    # subdirs = ['IBUG', 'LFPW', 'HELEN', 'AFW']
    subdirs = ['HELEN', 'AFW']
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        base_image_ids = get_base_image_ids(subdir_path)
        
        for base_image_id in tqdm(base_image_ids, desc=f"Processing {subdir}"):
            image_files = get_augmented_images(base_image_id, subdir_path)
            facymap = None
            for image_file in image_files:
                image_path = os.path.join(subdir_path, image_file)
                mat_path = image_path.replace('.jpg', '.mat')
                facymap = run_facymap_300W_LP(bfm, image_path, mat_path, save_folder, facymap, uv_h, uv_w, image_h, image_w)

if __name__ == '__main__':
    dataset_dir = 'Data/300W_LP'
    save_folder = 'results/facymap_300WLP_4995'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Set parameters
    uv_h = uv_w = 512
    image_h = image_w = 512

    # Load UV coordinates and BFM model
    uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/Out/BFM_UV.mat')
    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    bfm = MorphabelModel('Data/BFM/Out/BFM.mat')

    # Process the dataset
    process_dataset(bfm, dataset_dir, save_folder, uv_coords, uv_h, uv_w, image_h, image_w)

