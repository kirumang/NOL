import sys,os
sys.path.append(os.path.abspath('.'))  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import transforms3d as tf3d
import matplotlib.pyplot as plt
import time
import random
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow_graphics as tfg
from tensorflow.keras.applications.densenet import preprocess_input


from NOL_model import NOL_network as NOLnet
from NOL_model.NOL_network import simple_mask_error,iou_error,so3_to_mat
import NOL_tools.operations as to
#Rendering process 
import subprocess
import cv2
import NOL_tools.train_io as tio
from NOL_tools.train_io import data_generator



import math
from math import radians
from skimage.transform import resize
from skimage.morphology import binary_dilation
import h5py
import json


class NOL:
    def __init__(self,src_fn,im_h,im_w,cam_K,n_source=6):       
        train_model,backbone,render_gan,Tuning_pose =\
        NOLnet.neural_obj_learning(render_im_w=im_w,render_im_h=im_h,target_im_w=256,target_im_h=256,
                                   cam_K=cam_K)

        for layer in backbone.layers:
            layer.trainable=False  
        weight_fn ="./weights/nol_weights.hdf5"
        train_model.load_weights(weight_fn)
        print("Loading trained model from ",weight_fn)
        for layer in backbone.layers:
            layer.trainable=True

        gradient = K.gradients(Tuning_pose.outputs[0], [Tuning_pose.inputs[4],Tuning_pose.inputs[2]])
        iterate = keras.backend.function(Tuning_pose.inputs, [gradient,Tuning_pose.outputs[0],Tuning_pose.output[3]])

        simple_renderer = NOLnet.simple_render(img_h=im_h,img_w=im_w,cam_K=cam_K) 


        self.iterate = iterate
        self.render_gan = render_gan
        self.simple_renderer = simple_renderer      
        self.im_height=im_h
        self.im_width=im_w
        self.cam_K=cam_K          
        self.t_images,self.t_vert,self.t_faces,self.t_poses,self.t_bboxes,\
        self.t_bboxes_ori,self.gt_masks_full,self.gt_masks=self.load_data(src_fn)
        
        self.n_src = self.t_images.shape[0]        
        
        self.t_vert_vis=np.zeros((1000),bool)        
        
        self.t_face_angles=[]
        self.face_angles_full=[]        
        self.face_o3d = []        
        self.n_source = n_source
        print("Compute visible vertices in each poses...")
        self.update_vert_vis()
    
    def get_edge_imgage(self,img_source,render_xyz,c_box,th=0.1):
        render_xyz = render_xyz[c_box[0]:c_box[2],c_box[1]:c_box[3]]
        mask = render_xyz[:,:,2]>0
        dist_x,dist_y = np.gradient(render_xyz[:,:,2])
        depth_edge = np.logical_or(np.abs(dist_x)>th,np.abs(dist_y)>th)
        edge_mask = depth_edge
        edge_mask = resize(edge_mask.astype(np.float32),(256,256))>0
        edge_mask = binary_dilation(edge_mask)>0
        img_source[edge_mask>0]=[0,1,1]
        return img_source
    def display_inputs(self):
        f,ax = plt.subplots(1,self.n_src,figsize=(10,20))
        for fig_id in range(0,self.n_src):      
            simple_xyz = self.simple_renderer.predict([np.array([self.t_vert]),np.array([self.t_faces]),self.t_poses[fig_id:fig_id+1]])
            c_box = self.t_bboxes_ori[fig_id]
            render_xyz = simple_xyz[0][:,:,1:4] 
            img_source = np.copy(self.t_images[fig_id])
            if not(c_box[2]-c_box[0]<10 or c_box[3]-c_box[1]<10):
                img_source = self.get_edge_imgage(img_source,render_xyz,c_box,th=0.1)
            ax[fig_id].imshow(img_source)
    def get_updates(self,dx, x,v,iterations,momentum=0.9,lr=0.001,decay=0.99):
        '''
        grad:gradient
        params: input variable
        v = mu * v - learning_rate * dx # integrate velocity
        x += v # integrate position
        '''    
        if decay > 0:
            lr = lr *(decay**iterations) # (1. / (1. +decay ** iterations))
        v = momentum*v - lr*dx
        x += v
        return x,v
    def load_data(self,fn):        
        train_data = h5py.File(fn, "r")
        t_vert = np.array(train_data["vertices_3d"])
        t_faces =  np.array(train_data['faces'])
        
        t_images =  np.array(train_data['images'])
        t_poses = np.array(train_data['poses'])
        t_masks = np.array(train_data['masks'])        
        t_bboxes = np.array(train_data['bboxes']).astype(np.float32)
        t_bboxes_ori = np.copy(t_bboxes)
        t_bboxes[:,0]=t_bboxes[:,0]/(self.im_height-1)
        t_bboxes[:,1]=t_bboxes[:,1]/(self.im_width-1)
        t_bboxes[:,2]=(t_bboxes[:,2]-1)/(self.im_height-1)
        t_bboxes[:,3]=(t_bboxes[:,3]-1)/(self.im_width-1)
        t_bboxes = np.clip(t_bboxes,0,1)        

        gt_masks_full=[]
        gt_masks=[]
        t_bboxes_ori = t_bboxes_ori.astype(np.int)
        for im_id in range(t_images.shape[0]):
            mask = resize(t_masks[im_id].astype(np.float32),(t_bboxes_ori[im_id,2]-t_bboxes_ori[im_id,0],t_bboxes_ori[im_id,3]-t_bboxes_ori[im_id,1]))
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            full_mask = np.zeros((480,640,1))
            crop_mask = np.zeros((256,256,1))
            crop_mask[:,:,0:1]= t_masks[im_id]>0
            full_mask[t_bboxes_ori[im_id,0]:t_bboxes_ori[im_id,2],t_bboxes_ori[im_id,1]:t_bboxes_ori[im_id,3],0:1]=mask
            gt_masks_full.append(full_mask)
            gt_masks.append(crop_mask)    
        gt_masks_full = np.array(gt_masks_full)
        gt_masks = np.array(gt_masks)                
        train_data.close()
        return t_images,t_vert,t_faces,t_poses,t_bboxes,\
               t_bboxes_ori,gt_masks_full,gt_masks
        
    def update_vert_vis(self):
        c_vert = np.array(self.t_vert)
        face_obj = np.array(self.t_faces)
        #t_face_angles
        face_angles=[]
        face_angles_full=[]
        vert_visb=[]
        for im_id in range(self.n_src):
            pose = self.t_poses[im_id]
            vert_vis = np.ones((c_vert.shape[0]))    
            tf_vert = (np.matmul(pose[:3,:3],c_vert.T)+pose[:3,3:4])
            mesh_vert = np.copy(tf_vert.T)    
            mesh_normal = to.compute_face_normals(mesh_vert,face_obj)    
            bbox_new = self.t_bboxes_ori[im_id]
            
            simple_xyz = self.simple_renderer.predict([np.array([c_vert]),np.array([face_obj]),np.array([pose])])
            render_xyz = simple_xyz[0][:,:,1:4] 
            depth_r = render_xyz[:,:,2]
            xyz = to.getXYZ(depth_r,self.cam_K[0,0],self.cam_K[1,1],self.cam_K[0,2],self.cam_K[1,2])
            normal2,_ = to.get_normal(render_xyz[:,:,2],self.cam_K[0,0],self.cam_K[1,1],self.cam_K[0,2],self.cam_K[1,2])
            mask_effect = depth_r>0 #gt_masks_full[im_id,:,:,0]>0
            mask = depth_r>0
            normal2[np.invert(mask_effect)]=0
            camera_angle = np.copy(xyz)
            face_angle2= np.zeros( (self.im_height,self.im_width))
            valid_xyz = np.linalg.norm(camera_angle,axis=2)
            valid_normal = np.linalg.norm(normal2,axis=2)
            mask_face = np.logical_and(np.logical_and(depth_r>0,valid_xyz>0),valid_normal>0)
            face_angle2[mask_face] = np.sum(camera_angle[mask_face]*normal2[mask_face],axis=1)\
                                            /valid_xyz[mask_face]/valid_normal[mask_face]

            valid_indice = np.zeros((mesh_vert.shape[0]),bool)
            u_temp  =(self.cam_K[0,0]*tf_vert[0]/tf_vert[2] + self.cam_K[0,2]).astype(np.int)
            v_temp  =(self.cam_K[1,1]*tf_vert[1]/tf_vert[2] + self.cam_K[1,2]).astype(np.int)
            valid_idx = np.logical_and(u_temp>=0, u_temp+1<self.im_width)
            valid_idx = np.logical_and(np.logical_and(v_temp>=0, v_temp+1<self.im_height),valid_idx)    
            
            valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx],u_temp[valid_idx]]-tf_vert[2,valid_idx])<0.01
            valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx]+1,u_temp[valid_idx]]-tf_vert[2,valid_idx])<0.01
            valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx],u_temp[valid_idx]+1]-tf_vert[2,valid_idx])<0.01
            valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx]+1,u_temp[valid_idx]+1]-tf_vert[2,valid_idx])<0.01
            
            valid_uvs = np.zeros_like(valid_indice)
            
            #valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],1 - self.gt_masks_full[im_id][v_temp[valid_idx],u_temp[valid_idx],1])
            #valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],1 - self.gt_masks_full[im_id][v_temp[valid_idx]+1,u_temp[valid_idx],1])
            #valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],1 - self.gt_masks_full[im_id][v_temp[valid_idx],u_temp[valid_idx]+1,1])
            #valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],1 - self.gt_masks_full[im_id][v_temp[valid_idx]+1,u_temp[valid_idx]+1,1])   
            
            
            valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],self.gt_masks_full[im_id][v_temp[valid_idx],u_temp[valid_idx],0])
            valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],self.gt_masks_full[im_id][v_temp[valid_idx]+1,u_temp[valid_idx],0])
            valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],self.gt_masks_full[im_id][v_temp[valid_idx],u_temp[valid_idx]+1,0])
            valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],self.gt_masks_full[im_id][v_temp[valid_idx]+1,u_temp[valid_idx]+1,0])
            valid_indice[valid_idx] = np.logical_and(valid_indice[valid_idx],valid_uvs[valid_idx])   
            
            #remove invalid faces
            visible_face_id = (valid_indice[face_obj[:,0]]+valid_indice[face_obj[:,1]]+valid_indice[face_obj[:,2]])>0                            
            invalid_normal_face_id = face_obj[np.logical_and(mesh_normal[:,2]>0,
                                                             np.invert(visible_face_id))]
            valid_indice[invalid_normal_face_id[:,0]]=0
            valid_indice[invalid_normal_face_id[:,1]]=0
            valid_indice[invalid_normal_face_id[:,2]]=0            
        
            vert_visb.append(valid_indice)
            face_angles_full.append(face_angle2)
            face_angle_crop = resize(face_angle2[bbox_new[0]:bbox_new[2],bbox_new[1]:bbox_new[3]],(256,256))
            face_angles.append(face_angle_crop)
        self.t_vert_vis = np.array(vert_visb)
        self.t_face_angles  = np.array(face_angles)
        self.face_angles_full  = face_angles_full
        print("[Updated] Number of visible vertices:",np.sum(self.t_vert_vis,axis=1))                
    
    def get_batches(self,im_id,target_pose=np.array([0]),integrated=True):
        if(im_id==-1):
            target_pose = target_pose          
            pts = (np.matmul(target_pose[:3,:3],self.t_vert.T)+target_pose[:3,3:4]).T
            proj_u = self.cam_K[0,0]*pts[:,0]/pts[:,2]+self.cam_K[0,2]
            proj_v = self.cam_K[1,1]*pts[:,1]/pts[:,2]+self.cam_K[1,2]
            ct_u = int((np.max(proj_u)+np.min(proj_u))/2)
            ct_v = int((np.max(proj_v)+np.min(proj_v))/2)        
            bbox_ori = np.array([ct_v-128,ct_u-128,ct_v+128,ct_u+128])    
            bbox_ori[:2]=np.maximum(0,bbox_ori[:2])
            bbox_ori[2]=min(bbox_ori[2],self.im_height-1)
            bbox_ori[3]=min(bbox_ori[3],self.im_width-1)
        else:
            target_pose = self.t_poses[im_id]
            bbox_ori = self.t_bboxes_ori[im_id]
        
        t_masks =np.zeros((self.n_src,256,256,1))
        b_img,b_ang_in,b_vert3d,b_faces,\
        b_pose_input,b_vert_visible,b_bbox_train,\
        b_poses,b_bboxes,b_img_gt,uv_target,face_target,img_ids,\
        obj_3d\
        = to.get_tuning_set(target_pose,self.t_images,self.t_poses,self.t_bboxes,self.t_vert,self.t_faces,
                         self.simple_renderer,self.t_vert_vis,self.n_source,self.t_bboxes_ori,t_masks,self.t_face_angles,
                         cam_K=self.cam_K,res_x=self.im_width,res_y=self.im_height,bbox_ori=bbox_ori)
        if not(integrated):
            return b_img,b_ang_in,b_vert3d,b_faces,b_pose_input,\
                   b_vert_visible,b_bbox_train,b_poses,b_bboxes, \
                   self.gt_masks[np.array(img_ids).astype(int)],img_ids
        
        img_recont_norm,int_error,mask_output,proj_output,score_output,feature_in,uv_projection\
        = self.render_gan.predict([b_img,b_ang_in,b_vert3d,b_faces,
                              b_pose_input,b_vert_visible,b_bbox_train,
                              b_poses,b_bboxes])  
        proj_in = np.transpose(proj_output,[1,0,2,3,4]) #pose,batchxHxWxch
        score_output = np.transpose(score_output,[1,0,2,3,4]) #posexbatchxhxwx1
        img_sum = np.sum(score_output*proj_in,axis=1)
        init_img = tio.get_original_image(np.copy(img_sum[0,:,:,:3]))   
                
        mask_render = np.sum(mask_output[:,0,:,:,:1],axis=0,keepdims=True)>0
        f_gt_mask = self.gt_masks[np.array(img_ids).astype(int)]
        f_gt_masks = np.concatenate([mask_render,f_gt_mask],axis=0)        
        
        f_img,f_ang_in,f_vert3d,f_faces,f_pose_input,f_vert_visible,f_bbox_train,f_poses,f_bboxes =\
        to.add_integrated_img(img_sum[0][:,:,:3],face_target,target_pose,uv_target,bbox_ori,\
                           b_img,b_ang_in,b_vert3d,b_faces,b_pose_input,b_vert_visible,b_bbox_train,b_poses,b_bboxes)
        return f_img,f_ang_in,f_vert3d,f_faces,f_pose_input,\
               f_vert_visible,f_bbox_train,f_poses,f_bboxes,f_gt_masks,img_ids
    
    def iteration_pose_paper(self,f_img,f_ang_in,f_vert3d,f_faces,f_pose_input,\
                       f_vert_visible,f_bbox_train,f_poses,f_bboxes,f_gt_masks,img_ids,
                       max_iter,update_rate=1E-4):
        '''
        Original refinement code used for expriments in the paper
        '''
        update_rate = -update_rate
        prev_loss=-1     
        init_loss=-1
        actual_iter=0            
        while actual_iter<max_iter:    
            actual_iter+=1
            grad = self.iterate([f_img,f_ang_in,f_vert3d,f_faces,f_pose_input,f_vert_visible,f_bbox_train,
                            f_poses,f_bboxes])
            dgrad = grad[0][0]
            loss = grad[1][0]            
            if(np.abs(loss-prev_loss)<1E-6):
                break
            if(prev_loss!=-1 and prev_loss<loss):
                #if loss increased with the updates: recover and update with a smaller rate
                update_rate*=0.9 #decay update rate and recover old poses
                f_pose_input = np.copy(prev_pose_input)
                dgrad = np.copy(prev_dgrad)
            else:
                if(prev_loss==-1):init_loss=loss                    
                prev_loss=loss
                prev_pose_input = np.copy(f_pose_input)
                prev_dgrad = dgrad            
                update_rate*=0.99 #decay update rate               
            for fig_id in range(1,f_img.shape[0]):
                rx,ry,rz = tf3d.euler.mat2euler(dgrad[fig_id][:3,:3])
                dx,dy,dz =np.maximum(np.minimum(0.005,update_rate* dgrad[fig_id][:3,3]),-0.005)
                tf_adj = np.eye(4)
                tf_adj[:3,:3]=tf3d.euler.euler2mat(update_rate*rx,update_rate*ry,update_rate*rz)
                tf_adj[:3,3]=[dx,dy,dz]            
                f_pose_input[fig_id]=np.matmul(tf_adj,f_pose_input[fig_id])              
        return img_ids,f_pose_input[1:],loss,init_loss
        
    
    
    def render_a_pose(self,target_pose,**kwargs):
        #Sample source images for the target pose, and get an initial rendering
        f_img,f_ang_in,f_vert3d,f_faces,f_pose_input_so3,\
        f_vert_visible,f_bbox_train,f_poses,f_bboxes,f_gt_masks,img_ids=\
        self.get_batches(-1,target_pose)   

        #Local pose refinement for the target pose -> b_pose_input 
        img_ids,pose_new,loss,init_loss = self.iteration_pose_paper(f_img,f_ang_in,f_vert3d,f_faces,f_pose_input_so3,\
                                     f_vert_visible,f_bbox_train,f_poses,f_bboxes,f_gt_masks,img_ids,
                                     **kwargs)         
        #Rendering the final image with the updated pose
        img_recont_norm,int_error,mask_output,proj_output,score_output,feature_in,uv_projection\
        = self.render_gan.predict([f_img[1:],f_ang_in[1:],f_vert3d[1:],f_faces[1:],
                                  pose_new,f_vert_visible[1:],f_bbox_train[1:],
                                  f_poses[1:],f_bboxes[1:]])              
        img_sum = np.sum(score_output*proj_output,axis=0)
        result_img = tio.get_original_image(np.copy(img_sum[0,:,:,:3]))#normalized img to real img
        return result_img
    