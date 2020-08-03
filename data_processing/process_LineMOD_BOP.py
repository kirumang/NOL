'''
Sampling source images from LineMOD in BOP format (https://bop.felk.cvut.cz/)
usage: python3 data_processing/process_LineMOD_BOP.py [path/to/bop/lm] [obj_name=="obj_01/obj_02/..."]
The sample procedure is simpler since all images are fully visible. 
'''
#uniformly sample object crops using a renderer
import sys,os
sys.path.append(os.path.abspath('.'))  # To find local version of the library
sys.path.append(os.path.abspath('../'))  # To find local version of the library
import cv2
import numpy as np
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from skimage.transform import resize
import h5py
import math
from skimage.transform import resize
import NOL_tools.operations as to
import shutil
import transforms3d as tf3d
import json


if(len(sys.argv)<3):
    print("usage: python3 data_processing/process_LineMOD_BOP.py [path/to/bop/lm] [obj_name e.g., obj_01,obj_02,...]")
    sys.exit()
    
dataset_dir=sys.argv[1]    
t_label=sys.argv[2]
   
camera_intrinsic_fn = "./cfg_camera/camera_LineMOD.json"
with open(camera_intrinsic_fn, 'r') as f:
        camera_intrinsic = json.load(f)
cam_K=np.array(camera_intrinsic['cam_K']).reshape(3,3)
fx = cam_K[0,0]
fy = cam_K[1,1]
cx = cam_K[0,2]
cy = cam_K[1,2]
im_width = camera_intrinsic['width']
im_height = camera_intrinsic['height']
depth_th = 0.05

#Read GTS
test_dir = os.path.join(dataset_dir,"test/{:06d}".format(int(t_label[-2:])))
gt_fn = os.path.join(test_dir,"scene_gt.json")    
with open(gt_fn, 'r') as f:
    scene_gt_bop = json.load(f)

gt_info_fn = os.path.join(test_dir,"scene_gt_info.json")    
with open(gt_info_fn, 'r') as f:
    scene_gt_info_bop = json.load(f)


print("Loading the 3D model")
ply_fn = os.path.join(dataset_dir,"models/obj_{:06d}.ply".format(int(t_label[-2:])))
mesh = o3d.io.read_triangle_mesh(ply_fn)
vertices =np.array(mesh.vertices)/1000.0 #change to m scale (BOP models are in mm scale)
faces = np.array(mesh.triangles)


print("Images are sampled from the typical 15% split")
training_split = np.loadtxt("./data_processing/linemod_splits/obj_{:06d}.txt".format(int(t_label[-2:]))).astype(int)
n_scenes= training_split.shape[0]

img_cnt=0

poses=[]
visible_verts=[]
v_ids=[]
scores=[] 


for s_id in training_split:    
    scene_id = int(s_id)
    gts= scene_gt_bop[str(scene_id)]        
    gt_infos = scene_gt_info_bop[str(scene_id)]
    rgb_path = test_dir+"/rgb/{:06d}.png".format(scene_id)
    depth_path = test_dir+"/depth/{:06d}.png".format(scene_id) 
    

    #img = cv2.imread(rgb_path)[:,:,::-1] #rgb is not necessary this time
    depth = cv2.imread(depth_path,cv2.CV_16UC1)/1000                
    
    has_obj=False
    gt_id_=-1
    for gt_id,gt in enumerate(gts):
        obj_id = gt['obj_id']
        if(int(obj_id)==int(t_label[-2:])):
            tf_mat = np.eye(4)
            tf_mat[:3,:3] =np.array(gt['cam_R_m2c']).reshape(3,3)
            tf_mat[:3,3]  =np.array(gt['cam_t_m2c'])/1000
            has_obj=True
            gt_id_=gt_id
            visib = gt_infos[gt_id]['visib_fract']
            scores.append(visib)
            poses.append(tf_mat)
            v_ids.append(s_id)            
            break

print("Sampling images using poses...")

scores=np.array(scores)
sorted_idx = np.argsort(scores).tolist()
th_trans = 0.3
th_rot = 45
selected_frames=[]
#10cm, 15 degrees
while len(sorted_idx)>0:
    idx = sorted_idx[-1] #the last one has the highest value
    score = scores[idx]
    pose_q = poses[idx]
    selected_frames.append(v_ids[idx])
    del sorted_idx[-1]    
    merge_idx =[]
    for del_id,idx_c in enumerate(sorted_idx):
        pose_c = poses[idx_c]
        tra_diff = np.linalg.norm(pose_c[:3,3]-pose_q[:3,3])
        if(tra_diff<th_trans):
            rot_diff = np.abs(np.degrees(np.array(tf3d.euler.mat2euler(np.matmul( np.linalg.inv(pose_c[:3,:3]),pose_q[:3,:3])))))            
            if(rot_diff[0]<th_rot and rot_diff[1]<th_rot and rot_diff[2]<th_rot): #consider the flipped hand                
                merge_idx.append(del_id)                    
    for idx_c in range(len(merge_idx)-1,-1,-1):
        del sorted_idx[merge_idx[idx_c]]

print("No. selected frames:",len(selected_frames))
print("Selected frames:",selected_frames)


print("Save selected frames to the NOL source format (.hdf5)")

input_imgs=[]
poses_=[]
masks=[]
bboxes=[]
source_imgs=[]
for s_id in selected_frames:
    scene_id = int(s_id)
    gts= scene_gt_bop[str(scene_id)]        
    rgb_path = test_dir+"/rgb/{:06d}.png".format(scene_id)
    depth_path = test_dir+"/depth/{:06d}.png".format(scene_id) 

    img = cv2.imread(rgb_path)[:,:,::-1] #rgb is not necessary this time
    depth = cv2.imread(depth_path,cv2.CV_16UC1)/1000                
    
    has_obj=False
    gt_id_=-1
    for gt_id,gt in enumerate(gts):
        obj_id = gt['obj_id']
        if(int(obj_id)==int(t_label[-2:])):
            tf_mat = np.eye(4)
            tf_mat[:3,:3] =np.array(gt['cam_R_m2c']).reshape(3,3)
            tf_mat[:3,3]  =np.array(gt['cam_t_m2c'])/1000          
            has_obj=True
            gt_id_=gt_id
            break            
            
        
    #use pre_computed mask
    mask_fn = test_dir+"/mask/{:06d}_{:06d}.png".format(scene_id,gt_id_) 
    mask = np.sum(cv2.imread(mask_fn),axis=-1)>0
        
    #compute valid bbox and resize a cropped image and mask to 256,256
    
    img_masked = np.copy(img)
    img_masked = img_masked/255

    vu_list = np.where(mask)
    bbox = np.array([np.min(vu_list[0]),np.min(vu_list[1]),np.max(vu_list[0]),np.max(vu_list[1])],np.int32)
    height = bbox[2]-bbox[0]
    width = bbox[3]-bbox[1]
    ct_v =  int((bbox[2]+bbox[0])*0.5)
    ct_u =  int((bbox[3]+bbox[1])*0.5)
    
    length = int(max(max(height*0.5,width*0.5),128))
    img_ = np.zeros((256,256,3))
    mask_ = np.zeros((256,256,1))
    bbox_new = np.array([max(ct_v-length,0),max(ct_u-length,0),min(ct_v+length,im_height),min(ct_u+length,im_width)])
    
    img_crop = img_masked[bbox_new[0]:bbox_new[2],bbox_new[1]:bbox_new[3]]
    mask_crop = mask[bbox_new[0]:bbox_new[2],bbox_new[1]:bbox_new[3]]
    
    img_crop = resize(img_crop,(256,256))
    mask_crop = np.expand_dims(resize(mask_crop,(256,256))>0.5,axis=2)               
    
    
    input_imgs.append(img_crop)
    poses_.append(tf_mat)
    masks.append(mask_crop)  
    bboxes.append(bbox_new)
    
test_fn = "./sample_data/linemod/"+t_label+".hdf5"
if not(os.path.exists("./sample_data/linemod/")):os.makedirs("./sample_data/linemod/")    
train_data = h5py.File(test_fn, "w")
train_data.create_dataset("vertices_3d",data=vertices)
train_data.create_dataset("faces",data=np.array(faces))
train_data.create_dataset("images",data=np.array(input_imgs))
train_data.create_dataset("poses",data=np.array(poses_))
train_data.create_dataset("masks",data=np.array(masks))
train_data.create_dataset("bboxes",data=np.array(bboxes))
train_data.close() 
print("The source images for object:{} is successfully saved in ",test_fn)



