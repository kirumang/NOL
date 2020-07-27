'''
Example code that renders objects from uniform viewpoints in the upper-hemisphere.
The number of result iamges is 1296, which are used to train object recognizers in the evaluation
The NOL data file (.hdf5) must be created using one of the scripts in the "data_processing" folder.
'''
import os,sys
import numpy as np
import cv2
sys.path.append(os.path.abspath('.'))  
import tensorflow as tf
tf.autograph.set_verbosity(0)

from NOL_model import NOL
from matplotlib import pyplot as plt
import json

if(len(sys.argv)<3):
    print("usage: python3 examples/NOL_rendering.py [path/to/camera_cfg(.json)] [path/to/NOL_data(.hdf)] [(optional)target_dir]")
    sys.exit()

camera_cfg_fn=sys.argv[1]
src_fn = sys.argv[2]
if(len(sys.argv)==4):
    target_dir = sys.argv[3]
else:
    #default target : ./results/[data file name]/"
    filename= src_fn.split('/')[-1]
    filename = filename.replace(".hdf5","")    
    target_dir = "./results/{}".format(filename)
if not(os.path.exists(target_dir)): os.makedirs(target_dir)
    
with open(camera_cfg_fn, 'r') as f:
        camera_intrinsic = json.load(f)
cam_K=np.array(camera_intrinsic['cam_K']).reshape(3,3)
fx = cam_K[0,0]
fy = cam_K[1,1]
cx = cam_K[0,2]
cy = cam_K[1,2]
im_width = camera_intrinsic['width']
im_height = camera_intrinsic['height']


m_NOL = NOL.NOL(src_fn,im_h=im_height,im_w=im_width,cam_K=cam_K)


update_rate=1E-5 #for LineMOD and SMOT
pose_list = np.load("./sample_data/pose_target_lm.npy")
total_ids = np.arange(np.arange(0,360,5).shape[0]*np.arange(0,90,5).shape[0])
azi_ele = total_ids.reshape(72,18)
ele_azi = np.transpose(azi_ele).reshape(-1)
img_cnt=0

print("Start rendering the object...")


max_iter=50 #Decrease max_iter to increase the speed
back_color=[0,0,0] #background color (default:black, white:[1,1,1])
print("[Info] Decrease max_iter to increase the speed, current value:",max_iter)
for elaz_id in np.arange(0,ele_azi.shape[0],3):
    pose_id = ele_azi[elaz_id]
    pose = pose_list[pose_id]
    target_pose=np.eye(4)
    target_pose[:3,:3] =pose[:3,:3]
    target_pose[:3,3]=[0,0,0.7]
    img = m_NOL.render_a_pose(target_pose,max_iter=max_iter,update_rate=update_rate,back_color=[0,0,0]) #done
    img_cv2 = np.clip((img[:,:,::-1]*255).astype(np.uint8),0,255)
    cv2.imwrite(os.path.join(target_dir,"{:04d}.png".format(img_cnt)),img_cv2)
    print("Generated:",os.path.join(target_dir,"{:04d}.png".format(img_cnt)))
    img_cnt+=1    
    if(img_cnt%100==0):
        print("[Info] Decrease max_iter to increase the speed, current value:",max_iter)