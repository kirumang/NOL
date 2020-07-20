#uniformly sample object crops using a renderer
import sys,os
sys.path.append(os.path.abspath('.'))  # To find local version of the library
sys.path.append(os.path.abspath('../'))  # To find local version of the library
from NOL_model import NOL

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
import tools.operations as to
import shutil
import transforms3d as tf3d
import cpbd
import json

def render_depth(vertices,faces,pose,renderer):
    simple_xyz = renderer.predict([np.array([vertices]),
                                   np.array([faces]),
                                   np.array([pose])])
                           
    return simple_xyz[0][:,:,3]    
t_labels="obj_01" #sys.argv[1]
smot_dir ="/root/hdd_linux/SMOT_Upload" #sys.argv[2]

#Load Camera parameters
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

#Read GTS

#Define renderer
renderer = NOL.simple_render(img_h=im_height,img_w=im_width,cam_K=cam_K)


#Else?

for t_label in obj_labels:
    ply_fn = os.path.join(smot_dir+"/models",t_label+".ply")
    mesh = o3d.io.read_triangle_mesh(ply_fn)
    vertices_obj = np.array(mesh.vertices,np.float32)
    face_obj = np.array(mesh.triangles,np.int32)

    
    visratio = np.zeros((poses.shape[0]))   
    depth_th = 0.05
    occ_margin=0.01
    vis=False

    input_imgs=[]
    noisy_masks=[]
    vert_uvs=[]
    poses_=[]
    masks=[]
    after_frame=100
    frame_id=0


    #1-blurry score + visratio --> NMS via pose (remove all pose within 15 degree differences
    scores=[]
    v_ids=[]
    poses_cand=[]
    obj_tfs=[]
    visible_verts=[]
    for v_id in np.arange(0,poses.shape[0]-1,10):#mustard
        if(int(t_label[-2:])<5):
            rgb_fn = os.path.join(log_dir+"/color","{:06d}.jpg".format(v_id+1))
        else:
            rgb_fn = os.path.join(log_dir+"/color","{:06d}.png".format(v_id+1))
        depth_fn = os.path.join(log_dir+"/depth","{:06d}.png".format(v_id+1))
        mask_fn = os.path.join(log_dir+"/mask/"+t_label,"{:06d}.png".format(v_id+1))

        if not(os.path.exists(rgb_fn)):
            rgb_fn = os.path.join(log_dir+"/color","{:06d}.png".format(v_id+1))
            if not(os.path.exists(rgb_fn)):
                continue

        img = cv2.imread(rgb_fn)[:,:,::-1]
        depth = cv2.imread(depth_fn,cv2.CV_16UC1)/1000        
        obj_tf =obj_poses[v_id,1:].reshape(4,4)
        camera_pose = poses[v_id]
                
        ren.clear()
        ren.draw_model(obj_model,obj_tf)
        img_r, depth_r = ren.finish()
        img_r=img_r[:,:,::-1]        
        mask = np.logical_and(np.abs(depth_r-depth)<depth_th,depth_r>0)
        occ_mask = np.logical_and( (depth_r-depth)>=(depth_th+occ_margin),depth_r>0)
        #rendering 1m , depth=0.5 = 0.5
        mask_all = np.zeros_like(img)
        mask_all[mask]=[0,255,0] #green:visible mask
        mask_all[occ_mask]=[255,0,255] #blue:occ_mask
        
        if(np.sum(mask)<100):
            continue #skip when there is too small mask
        
        if True:#Calcualte visible vertex indices
            mesh_temp = copy.deepcopy(mesh)
            mesh_temp.transform(obj_tf)
            mesh_vert = np.array(mesh_temp.vertices)
            mesh_normal = compute_face_normals(mesh_vert, face_obj)
            xyz = to.getXYZ(depth_r,fx,fy,cx,cy)
            normal2,_ = to.get_normal(depth_r,fx,fy,cx,cy)
            normal2[np.invert(mask)]=0
            camera_angle = np.copy(xyz)
            face_angle2= np.zeros( (im_height,im_width))
            valid_xyz = np.linalg.norm(camera_angle,axis=2)
            valid_normal = np.linalg.norm(normal2,axis=2)
            mask_face = np.logical_and(np.logical_and(mask,valid_xyz>0),valid_normal>0)
            
            face_angle2[mask_face] = np.sum(camera_angle[mask_face]*normal2[mask_face],axis=1)/valid_xyz[mask_face]/valid_normal[mask_face]
            
            xyz = xyz[mask]
            nn = NearestNeighbors(n_neighbors=50).fit(mesh_vert)                    
            dists,indices = nn.kneighbors(xyz)
            vert_uv_img = np.zeros((vertices_obj.shape[0],2))
            vert_mask = np.zeros((mesh_vert.shape[0]),np.uint8)
            valid_indice = np.zeros((mesh_vert.shape[0]),bool)
            for pt_idx in range(xyz.shape[0]):
                valid_ver = indices[pt_idx,dists[pt_idx,:]<0.01]                
                vert_mask[valid_ver]=1            
            visible_face_id = (vert_mask[face_obj[:,0]]+vert_mask[face_obj[:,1]]+vert_mask[face_obj[:,2]])==3                        
            valid_normal_face_id = mesh_normal[:,2]<0
            valid_faces =face_obj[np.logical_and(visible_face_id,valid_normal_face_id)]
            valid_indice[valid_faces[:,0]]=1
            valid_indice[valid_faces[:,1]]=1
            valid_indice[valid_faces[:,2]]=1 
            visible_verts.append(valid_indice)
        #reproject to 3D points in mask
        ren.clear()
        obj_tf_center=np.copy(obj_tf)
        obj_tf_center[0,3]=0
        obj_tf_center[1,3]=0
        ren.draw_model(obj_model,obj_tf_center)
        _, depth_r_center = ren.finish()
        n_full = np.sum(depth_r_center>0)
        n_visible = np.sum(mask) 
        visratio[v_id]=n_visible/n_full
        sharp_score = 1#cpbd.compute(np.mean(img,axis=2))

        scores.append(sharp_score*n_visible/n_full)
        poses_cand.append(obj_tf)
        v_ids.append(v_id)
        #if(visratio[v_id]>0.90 and after_frame>10):
        #    print(cpbd.compute(img[:,:,0]))
        #    selected_frames.append(v_id)
        #    after_frame=0
            #eval less blurry images /
    scores=np.array(scores)
    sorted_idx = np.argsort(scores).tolist()
    th_trans = 0.3
    th_rot = 45
    selected_frames=[]
    #10cm, 15 degrees    
    visible_v =np.copy(np.array(visible_verts))
    sum_visible = np.zeros((visible_v.shape[1]))
    data_idx=np.arange(scores.shape[0])
    max_n=20
    while visible_v.shape[0]>0:
        active_idx = np.ones((visible_v.shape[0]),bool)    
        after_visible = sum_visible+visible_v
        vert_score = np.sum(np.tanh(1.5*after_visible),axis=1)
        idx =np.argmax(vert_score)    
        active_idx[idx]=0
        before =  np.sum(np.tanh(1.5*sum_visible))
        sum_visible+= visible_v[idx]
        after  =  np.sum(np.tanh(1.5*sum_visible))
        print(after-before)
        if(after-before<1):
            break

        d_idx = data_idx[idx]
        score = scores[d_idx]
        pose_q = poses_cand[d_idx]
        v_id = v_ids[d_idx]
        selected_frames.append(v_id)
        if(len(selected_frames)>max_n):
            break

        for del_id,idx_c in enumerate(data_idx):
            pose_c = poses_cand[idx_c]
            tra_diff = np.linalg.norm(pose_c[:3,3]-pose_q[:3,3])
            if(tra_diff<0.1):
                rot_diff = np.abs(np.degrees(np.array(tf3d.euler.mat2euler(np.matmul( np.linalg.inv(pose_c[:3,:3]),pose_q[:3,:3])))))            
                if(rot_diff[0]<15 and rot_diff[1]<15 and rot_diff[2]<15 ): #consider the flipped hand                
                    active_idx[del_id]=0

        print(np.sum(active_idx),active_idx.shape[0])    
        data_idx = data_idx[active_idx]
        visible_v = visible_v[active_idx]
        print(visible_v.shape[0],data_idx.shape[0])

    print("no of selected frames:",len(selected_frames))    

    
    input_imgs=[]
    vert_uvs=[]
    poses_=[]
    masks=[]
    bboxes=[]
    face_angles=[]
    frame_id=0
    source_imgs=[]
    for v_id in selected_frames:
        if(int(t_label[-2:])<5):
            rgb_fn = os.path.join(log_dir+"/color","{:06d}.jpg".format(v_id+1))
        else:
            rgb_fn = os.path.join(log_dir+"/color","{:06d}.png".format(v_id+1))
            
        depth_fn = os.path.join(log_dir+"/depth","{:06d}.png".format(v_id+1))
        mask_fn = os.path.join(log_dir+"/mask/"+t_label,"{:06d}.png".format(v_id+1))

        img = cv2.imread(rgb_fn)[:,:,::-1]
        depth = cv2.imread(depth_fn,cv2.CV_16UC1)/1000        
        obj_tf =obj_poses[v_id,1:].reshape(4,4)
        camera_pose = poses[v_id]
        #noICP
        
        points_tgt = np.zeros((depth.shape[0],depth.shape[1],6),np.float32)
        points_tgt[:,:,:3] = to.getXYZ(depth,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])
        points_tgt[:,:,3:],_ = to.get_normal(depth,fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])       
        ren.clear()
        ren.draw_model(obj_model,obj_tf)
        img_r1, depth_r = ren.finish()
        img_r1=img_r1[:,:,::-1]
        mask = np.logical_and(np.abs(depth_r-depth)<depth_th,depth_r>0)
        #ICP and re-rendering and obtaining mask        
        rot_pred = obj_tf[:3,:3]
        tra_pred = obj_tf[:3,3]
        pts_tgt = points_tgt[mask]        
        obj_tf,residual = icp_refinement(pts_tgt,obj_model,np.copy(rot_pred),tra_pred,cam_K,ren,mask)                
        
        if(t_label=="obj_06" or t_label=="obj_07"):
            obj_tf[:3,:3]=rot_pred
        #ignore rotation for obj_06,07
        
        ren.clear()
        ren.draw_model(obj_model,obj_tf)
        img_r, depth_r = ren.finish()
        img_r=img_r[:,:,::-1]
        mask = np.logical_and(np.abs(depth_r-depth)<depth_th,depth_r>0)
        occ_mask = np.logical_and( (depth_r-depth)>=(depth_th+occ_margin),depth_r>0)
        
        
        #calculate uv value
        mesh_temp = copy.deepcopy(mesh)
        mesh_temp.transform(obj_tf)
        mesh_vert = np.array(mesh_temp.vertices)
        mesh_normal = compute_face_normals(mesh_vert, face_obj)

        ren.clear()
        #obj_model.set_new_attribute(mesh_normal) 

        ren.draw_model(obj_model,obj_tf)
        _, depth_r = ren.finish()

        xyz = to.getXYZ(depth_r,fx,fy,cx,cy)
        normal2,_ = to.get_normal(depth_r,fx,fy,cx,cy)
        normal2[np.invert(mask)]=0
        camera_angle = np.copy(xyz)
        #dot prodcut
        #face_angle = np.sum(camera_angle*normal_pixel,axis=2) #dot product
        face_angle2= np.zeros( (im_height,im_width))
        valid_xyz = np.linalg.norm(camera_angle,axis=2)
        valid_normal = np.linalg.norm(normal2,axis=2)
        mask_face = np.logical_and(np.logical_and(mask,valid_xyz>0),valid_normal>0)
        face_angle2[mask_face] = np.sum(camera_angle[mask_face]*normal2[mask_face],axis=1)/valid_xyz[mask_face]/valid_normal[mask_face]

        #mask = depth_r>0
        xyz = xyz[mask]
        img_masked = np.copy(img)

        #img_masked[np.invert(mask)]=0        

        nn = NearestNeighbors(n_neighbors=50).fit(mesh_vert)                    
        dists,indices = nn.kneighbors(xyz)
        vert_uv_img = np.zeros((vertices_obj.shape[0],2))
        vert_mask = np.zeros((mesh_vert.shape[0]),np.uint8)
        valid_indice = np.zeros((mesh_vert.shape[0]),bool)
        for pt_idx in range(xyz.shape[0]):
            valid_ver = indices[pt_idx,dists[pt_idx,:]<0.01]                
            vert_mask[valid_ver]=1            
        visible_face_id = (vert_mask[face_obj[:,0]]+vert_mask[face_obj[:,1]]+vert_mask[face_obj[:,2]])==3                        
        valid_normal_face_id = mesh_normal[:,2]<0
        valid_faces =face_obj[np.logical_and(visible_face_id,valid_normal_face_id)]
        valid_indice[valid_faces[:,0]]=1
        valid_indice[valid_faces[:,1]]=1
        valid_indice[valid_faces[:,2]]=1  
        u_temp = fx*mesh_vert[valid_indice,0]/mesh_vert[valid_indice,2]+cx
        v_temp = fy*mesh_vert[valid_indice,1]/mesh_vert[valid_indice,2]+cy
        
        uv_valuable = mask[v_temp.astype(np.int),u_temp.astype(np.int)].astype(np.float32)
        
        vert_uv_img[valid_indice,0]=u_temp*uv_valuable
        vert_uv_img[valid_indice,1]=v_temp*uv_valuable
        #remove u_temp,v_temp if they are not surrounded by mask
        
        #crop the image region and resize to 256x256 if it is too large
        vu_list = np.where(mask)
        bbox = np.array([np.min(vu_list[0]),np.min(vu_list[1]),np.max(vu_list[0]),np.max(vu_list[1])],np.int32)
        height = bbox[2]-bbox[0]
        width = bbox[3]-bbox[1]
        ct_v =  int((bbox[2]+bbox[0])*0.5)
        ct_u =  int((bbox[3]+bbox[1])*0.5)
        img_masked = img_masked/255

        length = int(max(max(height*0.5,width*0.5),128))
        img_ = np.zeros((256,256,3))
        mask_ = np.zeros((256,256,1))
        #new_bboxes
        bbox_new = np.array([max(ct_v-length,0),max(ct_u-length,0),min(ct_v+length,im_height),min(ct_u+length,im_width)])
        vert_uv_img[valid_indice,0]  = (vert_uv_img[valid_indice,0] -bbox_new[1])/(bbox_new[3]-bbox_new[1]-1)
        vert_uv_img[valid_indice,1]  = (vert_uv_img[valid_indice,1] -bbox_new[0])/(bbox_new[2]-bbox_new[0]-1)
        vert_uv_img[np.logical_or(vert_uv_img<0,vert_uv_img>1)]=0
        
        img_masked = img_masked[bbox_new[0]:bbox_new[2],bbox_new[1]:bbox_new[3]]
        source_imgs.append(img_masked)
        
        mask = mask[bbox_new[0]:bbox_new[2],bbox_new[1]:bbox_new[3]]
        
        img_masked = resize(img_masked,(256,256))
        mask = np.expand_dims(resize(mask,(256,256))>0.5,axis=2)               
        face_angle2 = resize(face_angle2[bbox_new[0]:bbox_new[2],bbox_new[1]:bbox_new[3]],(256,256))
        img_masked[:2,:2]=0 #trick for zero uv values        


        input_imgs.append(img_masked)
        vert_uvs.append(vert_uv_img)
        poses_.append(obj_tf)
        masks.append(mask)  
        bboxes.append(bbox_new)
        face_angles.append(face_angle2)
        #color_00.cam,color_00.jpg,depth_91.png
        if(save_g2ltex):
            g2ltex_rgb = os.path.join(g2ltex_dir+"/textureimages","color_{:02d}.png".format(frame_id))
            g2ltex_cam = os.path.join(g2ltex_dir+"/textureimages","color_{:02d}.cam".format(frame_id))
            g2ltex_depth = os.path.join(g2ltex_dir+"/textureimages","depth_{:02d}.png".format(frame_id))            
            shutil.copyfile(rgb_fn, g2ltex_rgb) 
            shutil.copyfile(depth_fn, g2ltex_depth) 
            cam_pose = matpose(poses[v_id])      
            #obj_tf =obj_poses[v_id,1:].reshape(4,4)
            #camera_pose = poses[v_id]
        
            cam_pose = obj_poses[v_id,1:].reshape(4,4) #np.linalg.inv(cam_pose)  #np.linalg.inv(obj_tf) #np.linalg.inv(cam_pose)   
            cam_g2ltex = np.zeros((12))
            cam_g2ltex[:3]=cam_pose[:3,3]
            cam_g2ltex[3:] = np.reshape(cam_pose[:3,:3],-1)
            np.savetxt(g2ltex_cam,cam_g2ltex,fmt ='%.6f',newline=' ')
            
        if(save_mvs):        
            mvs_rgb = os.path.join(mvs_dir+"/textureimages","color_{:02d}.png".format(frame_id))
            mvs_cam = os.path.join(mvs_dir+"/textureimages","color_{:02d}.cam".format(frame_id))
            mvs_depth = os.path.join(mvs_dir+"/textureimages","depth_{:02d}.png".format(frame_id))
            shutil.copyfile(rgb_fn, mvs_rgb) 
            shutil.copyfile(depth_fn, mvs_depth)         
            
            flen =cam_K[0,0] / max(im_width, im_height)               
            cam_pose = obj_poses[v_id,1:].reshape(4,4) #matpose(poses[v_id])        
            #cam_pose = np.linalg.inv(cam_pose)           
            cam_g2ltex = np.zeros((12))
            cam_g2ltex[:3]=cam_pose[:3,3]
            cam_g2ltex[3:] = np.reshape(cam_pose[:3,:3],-1)

            ext = cam_g2ltex
            cam_file_mvs = open(mvs_cam,'w')
            cam_file_mvs.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                     ext[0],ext[1],ext[2],ext[3],ext[4],ext[5],ext[6],ext[7],ext[8],ext[9],ext[10],ext[11]))
            cam_file_mvs.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                     flen,0,0,1.0,cam_K[0,2]/(im_width-1),cam_K[1,2]/(im_height-1)))
            cam_file_mvs.close()
        
        frame_id+=1
        if(vis):
            f,ax = plt.subplots(1,3,figsize=(20,20))
            ax[0].imshow(img_r)
            ax[1].imshow(mask_all)
            ax[2].imshow(depth_r)            
        after_frame=0
    if(save_nom):
        test_fn = test_dir+t_label+".hdf5"
        if not(os.path.exists(test_dir)):os.makedirs(test_dir)    
        train_data = h5py.File(test_fn, "w")
        #to save: vertcies(Nx3) , uv_map (Nx2), faces (Fx3), images(10,256,256,3),masks, gt_texture(wxhx3)
        train_data.create_dataset("vertices_3d",data=vertices_obj)
        train_data.create_dataset("vertices_uv",data=np.array(vert_uvs))
        train_data.create_dataset("faces",data=np.array(face_obj))
        train_data.create_dataset("images",data=np.array(input_imgs))
        train_data.create_dataset("poses",data=np.array(poses_))
        train_data.create_dataset("masks",data=np.array(masks))
        train_data.create_dataset("bboxes",data=np.array(bboxes))
        train_data.create_dataset("face_angles",data=np.array(face_angles))
        train_data.close() 

        
#source_imgs.append(img_masked)     im   
n_source = len(source_imgs)
f,ax= plt.subplots(1,n_source,figsize=(n_source*10,20))
for i in range(n_source):
    ax[i].imshow(source_imgs[i])
