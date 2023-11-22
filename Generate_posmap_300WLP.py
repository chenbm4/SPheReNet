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

def run_posmap_300W_LP(bfm, image_path, mat_path, save_folder,  uv_h = 256, uv_w = 256, image_h = 256, image_w = 256):
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

    # 4. uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c = 3)

    cropped_image = np.clip(cropped_image, 0, 1)
    cropped_image_8bit = (cropped_image * 255).astype(np.uint8)
    # 5. save files
    io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image_8bit))
    np.save('{}/{}'.format(save_folder, image_name.replace('jpg', 'npy')), uv_position_map)
    uv_position_map = (uv_position_map)/max(image_h, image_w)
    uv_position_map = np.clip(uv_position_map, 0, 1)
    uv_position_map_8bit = (uv_position_map * 255).astype(np.uint8)
    io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_posmap.jpg')), uv_position_map_8bit) # only for show

    # --verify
    # import cv2
    # uv_texture_map_rec = cv2.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_tex.jpg')), np.squeeze(uv_texture_map_rec))


def process_dataset(bfm, dataset_dir, save_folder, uv_coords, uv_h=256, uv_w=256, image_h=256, image_w=256):
    subdirs = ['IBUG', 'LFPW', 'HELEN', 'AFW']
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        image_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
        
        for image_file in image_files:
            image_path = os.path.join(subdir_path, image_file)
            mat_path = image_path.replace('.jpg', '.mat')
            run_posmap_300W_LP(bfm, image_path, mat_path, save_folder, uv_h, uv_w, image_h, image_w)

if __name__ == '__main__':
    dataset_dir = 'Data/300W_LP'
    save_folder = 'results/posmap_300WLP_4995'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Set parameters
    uv_h = uv_w = 256
    image_h = image_w = 256

    # Load UV coordinates and BFM model
    uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/Out/BFM_UV.mat')
    uv_coords = process_uv(uv_coords, uv_h, uv_w)
    bfm = MorphabelModel('Data/BFM/Out/BFM.mat')

    # Process the dataset
    process_dataset(bfm, dataset_dir, save_folder, uv_coords, uv_h, uv_w, image_h, image_w)

