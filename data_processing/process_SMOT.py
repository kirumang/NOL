'''
Sampling source images from SMOT dataset
Download: https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/smot/
usage: python3 data_processing/process_SMOT.py [path/to/smot] [obj_name=="obj_01/obj_02/..."]
'''
#uniformly sample object crops using a renderer
import sys,os
sys.path.append(os.path.abspath('.'))  # To find local version of the library
sys.path.append(os.path.abspath('../'))  # To find local version of the library
from NOL_model import NOL_network as NOLnet
import cv2
import yaml
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
    print("usage: python3 data_processing/process_SMOT.py [path/to/smot] [obj_name e.g., obj_01,obj_02,...] [icp=yes(1)/no(0),default=1]")
    sys.exit()
    
#smot_dir ="/root/hdd_linux/SMOT_Upload" #sys.argv[1]
#t_label="obj_01"

smot_dir=sys.argv[1]    
t_label=sys.argv[2]

icp=1
if(len(sys.argv)==4):
    icp=int(sys.argv[3])
   

#Load Camera parameters of SMOT dataset
camera_intrinsic_fn = os.path.join(smot_dir,"camera_intrinsic.json")
with open(camera_intrinsic_fn, 'r') as f:
        camera_intrinsic = json.load(f)
cam_K=np.array(camera_intrinsic['intrinsic_matrix']).reshape(3,3).T
fx = cam_K[0,0]
fy = cam_K[1,1]
cx = cam_K[0,2]
cy = cam_K[1,2]
im_width = camera_intrinsic['width']
im_height = camera_intrinsic['height']
depth_th = 0.05 #depth threshold for visible mask

#Read GTS
d_type='train'
if(int(t_label[-2:])<5): 
    #obj_01~04: sequence 1
    test_dir = os.path.join(smot_dir,d_type+"/seq{}".format(1))    
else:
    #obj_05~08: sequence 2
    test_dir = os.path.join(smot_dir,d_type+"/seq{}".format(2))    

gt_fn = os.path.join(test_dir,"scene_gt.json")    
with open(gt_fn, 'r') as f:
    scene_gt_smot = json.load(f)

#Define renderer
renderer = NOLnet.simple_render(img_h=im_height,img_w=im_width,cam_K=cam_K)


print("Loading the 3D model")
ply_fn = os.path.join(smot_dir,"models/{}.ply".format(t_label))
mesh = o3d.io.read_triangle_mesh(ply_fn)
vertices =np.array(mesh.vertices)
faces = np.array(mesh.triangles)


print("Computing visible verticies in each image")
n_scenes= len(scene_gt_smot.keys())
img_cnt=0

poses=[]
visible_verts=[]
v_ids=[]
scores=[] #visibility score to select the best image (with less occlusion) at the beginning

for s_id in sorted(scene_gt_smot.keys()):    
    scene_id = int(s_id)
    gts= scene_gt_smot[str(scene_id)]        
    rgb_path = test_dir+"/color/{:06d}.png".format(scene_id)
    depth_path = test_dir+"/depth/{:06d}.png".format(scene_id) 

    #img = cv2.imread(rgb_path)[:,:,::-1] #rgb is not necessary this time
    depth = cv2.imread(depth_path,cv2.CV_16UC1)/1000                
    
    has_obj=False
    for gt in gts:
        obj_id = gt['obj_id']
        if(int(obj_id)==int(t_label[-2:])):
            tf_mat = np.eye(4)
            tf_mat[:3,:3] =np.array(gt['cam_R_m2c']).reshape(3,3)
            tf_mat[:3,3]  =np.array(gt['cam_t_m2c'])            
            has_obj=True
            break
    
    if(has_obj):
        simple_xyz = renderer.predict([np.array([vertices]),np.array([faces]),
                                               np.array([tf_mat])])[0]
        
        depth_r = simple_xyz[:,:,3]
        mask = np.logical_and(np.abs(depth_r-depth)<depth_th,depth_r>0)
        
        valid_vertices = to.get_valid_vertices(vertices,faces,depth_r,tf_mat,cam_K,im_width,im_height,
                                               mask=mask)
        
        n_full = np.sum(depth_r>0)
        n_visible = np.sum(mask) 
        visratio=n_visible/n_full
        scores.append(n_visible/n_full)
        poses.append(tf_mat)
        v_ids.append(s_id)
        visible_verts.append(valid_vertices)
        
        
    if(len(v_ids)%10==0):
        print("processing: {:04d}/{:04d}".format(len(v_ids),n_scenes))

print("Finished:Computing visible verticies")    



print("Iteratively adding new source images...")
scores=np.array(scores)
sorted_idx = np.argsort(scores).tolist()
th_trans = 0.3
th_rot = 45
selected_frames=[]

visible_v =np.copy(np.array(visible_verts))
sum_visible = np.zeros((visible_v.shape[1]))
data_idx=np.arange(scores.shape[0])

max_n=15 #limit maximum number of images

while visible_v.shape[0]>0:
    active_idx = np.ones((visible_v.shape[0]),bool)    
    after_visible = sum_visible+visible_v
    vert_score = np.sum(np.tanh(1.5*after_visible),axis=1)
    idx =np.argmax(vert_score)    
    active_idx[idx]=0
    before =  np.sum(np.tanh(1.5*sum_visible))
    sum_visible+= visible_v[idx]
    after  =  np.sum(np.tanh(1.5*sum_visible))
    print("Increased score for the observed vertices:", after-before)
    if(after-before<1): #terminate the improvement is less than 1
        break

    d_idx = data_idx[idx]
    score = scores[d_idx]
    pose_q = poses[d_idx]
    v_id = v_ids[d_idx]
    selected_frames.append(v_id)
    if(len(selected_frames)>max_n):
        break

    for del_id,idx_c in enumerate(data_idx):
        pose_c = poses[idx_c]
        tra_diff = np.linalg.norm(pose_c[:3,3]-pose_q[:3,3])
        if(tra_diff<0.1):
            rot_diff = np.abs(np.degrees(np.array(tf3d.euler.mat2euler(np.matmul( np.linalg.inv(pose_c[:3,:3]),pose_q[:3,:3])))))            
            if(rot_diff[0]<15 and rot_diff[1]<15 and rot_diff[2]<15 ): #consider the flipped hand                
                active_idx[del_id]=0
    data_idx = data_idx[active_idx]
    visible_v = visible_v[active_idx]
    
print("No. selected frames:",len(selected_frames))
print("Selected frames:",selected_frames)


print("Save selected frames to the NOL source format (.hdf5)")
if(icp==1):
    print("ICP is enabled")
else:
    print("[INFO] It is possible to perform ICP to have better poses using depth images, but it sometimes produces worse poses")

    
input_imgs=[]
poses_=[]
masks=[]
bboxes=[]
source_imgs=[]
for s_id in selected_frames:
    scene_id = int(s_id)
    gts= scene_gt_smot[str(scene_id)]        
    rgb_path = test_dir+"/color/{:06d}.png".format(scene_id)
    depth_path = test_dir+"/depth/{:06d}.png".format(scene_id) 

    img = cv2.imread(rgb_path)[:,:,::-1] #rgb is not necessary this time
    depth = cv2.imread(depth_path,cv2.CV_16UC1)/1000                
    
    has_obj=False
    for gt in gts:
        obj_id = gt['obj_id']
        if(int(obj_id)==int(t_label[-2:])):
            tf_mat = np.eye(4)
            tf_mat[:3,:3] =np.array(gt['cam_R_m2c']).reshape(3,3)
            tf_mat[:3,3]  =np.array(gt['cam_t_m2c'])            
            has_obj=True
            break            
            
    #noICP
    simple_xyz = renderer.predict([np.array([vertices]),np.array([faces]),
                                               np.array([tf_mat])])[0]
    depth_r = simple_xyz[:,:,3]
    mask = np.logical_and(np.abs(depth_r-depth)<depth_th,depth_r>0)
    if(icp==1):        
        points_src = np.zeros((im_height,im_width,6),np.float32)
        points_src[:,:,:3] = to.getXYZ(depth_r,cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2])
        points_src[:,:,3:],_ = to.get_normal(depth_r,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])
        points_src = points_src[mask]
        
        points_tgt = np.zeros((depth.shape[0],depth.shape[1],6),np.float32)
        points_tgt[:,:,:3] = to.getXYZ(depth,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])
        points_tgt[:,:,3:],_ = to.get_normal(depth,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])     
        
        pts_tgt = points_tgt[mask]
        
        icp_fnc = cv2.ppf_match_3d_ICP(100,tolerence=0.05,numLevels=4) #1cm
        retval, residual, pose=icp_fnc.registerModelToScene(points_src.reshape(-1,6), pts_tgt.reshape(-1,6))        
        tf_mat = np.matmul(pose,tf_mat)  
        simple_xyz = renderer.predict([np.array([vertices]),np.array([faces]),
                                               np.array([tf_mat])])[0]
        depth_r = simple_xyz[:,:,3]
        mask = np.logical_and(np.abs(depth_r-depth)<depth_th,depth_r>0)

    
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
    
test_fn = "./sample_data/smot/"+t_label+".hdf5"
if not(os.path.exists("./sample_data/smot/")):os.makedirs("./sample_data/smot/")    
train_data = h5py.File(test_fn, "w")
train_data.create_dataset("vertices_3d",data=vertices)
train_data.create_dataset("faces",data=np.array(faces))
train_data.create_dataset("images",data=np.array(input_imgs))
train_data.create_dataset("poses",data=np.array(poses_))
train_data.create_dataset("masks",data=np.array(masks))
train_data.create_dataset("bboxes",data=np.array(bboxes))
train_data.close() 
print("The source images for object:{} is successfully saved in ",test_fn)


