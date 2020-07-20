import os,sys
import numpy as np
import cv2
sys.path.append(os.path.abspath('.'))  
tf.autograph.set_verbosity(0)

from NOL_model import NOL

'''
Todo: replace im_w,im_h, with a config file
Arg1: config.json
Arg2: src_fn

'''

#src_fn,im_h,im_w,cam_K):  
dataset="ho3d"
if(dataset=="smot"):
    cam_K = np.array([[538.391033533567,0,315.3074696331638],
                      [0,538.085452058436,233.0483557773859],
                      [0,0,1]])
    fx = cam_K[0,0]
    fy = cam_K[1,1]
    cx = cam_K[0,2]
    cy = cam_K[1,2]
    im_h=480
    im_w=640

else:
    im_w = 640
    im_h = 480
    cam_K=np.array([[614.0,   0.   , 639/2],
                     [  0.   , 614.0, 479/2],
                     [  0.   ,   0.   ,   1. ]])

src_fn = "./sample_data/ho3d/010_potted_meat_can.hdf5"
m_NOL = NOL.NOL(src_fn,im_h=im_h,im_w=im_w,cam_K=cam_K)


result_dir = "./results"
if not(os.path.exists(result_dir)):os.makedirs(result_dir)
    
update_rate=1E-5 #for LineMOD and SMOT
pose_list = np.load("./sample_data/pose_target_lm.npy")
total_ids = np.arange(np.arange(0,360,5).shape[0]*np.arange(0,90,5).shape[0])
azi_ele = total_ids.reshape(72,18)
ele_azi = np.transpose(azi_ele).reshape(-1)
img_cnt=0
for elaz_id in np.arange(0,ele_azi.shape[0],3):#[target_pose_id]:
    pose_id = ele_azi[elaz_id]
    pose = pose_list[pose_id]
    target_pose=np.eye(4)
    target_pose[:3,:3] =pose[:3,:3]
    target_pose[:3,3]=[0,0,0.7]
    img = m_NOL.render_a_pose(target_pose,max_iter=50,update_rate=update_rate) #done
    img_cv2 = np.clip((img[:,:,::-1]*255).astype(np.uint8),0,255)
    cv2.imwrite(os.path.join(result_dir,"{:04d}.png".format(img_cnt)),img_cv2)
    img_cnt+=1
