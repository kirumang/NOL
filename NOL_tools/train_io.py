import sys,os
import numpy as np
import scipy.ndimage
import math
import random
import h5py
from imgaug import augmenters as iaa
from skimage.transform import resize
import cv2
from skimage import morphology
from tensorflow.keras.applications.densenet import preprocess_input
import transforms3d as tf3d

def get_a_test_data(fn):
    train_data = h5py.File(fn, "r")
    t_vert = np.array(train_data["vertices_3d"])
    t_vert_uv = np.array(train_data["vertices_uv"])
    t_faces =  np.array(train_data['faces'])
    t_images =  np.array(train_data['images'])
    t_poses = np.array(train_data['poses'])
    t_masks = np.array(train_data['masks'])
    t_face_angles = np.array(train_data['face_angles'])
    t_bboxes = np.array(train_data['bboxes']).astype(np.float32)
    t_bboxes_ori = np.copy(t_bboxes)
    t_bboxes[:,0]=t_bboxes[:,0]/(480-1)
    t_bboxes[:,1]=t_bboxes[:,1]/(640-1)
    t_bboxes[:,2]=(t_bboxes[:,2]-1)/(480-1)
    t_bboxes[:,3]=(t_bboxes[:,3]-1)/(640-1)
    t_bboxes = np.clip(t_bboxes,0,1)

    
    train_data.close()  
    im_h = t_images.shape[1]
    im_w = t_images.shape[2]
    n_image= t_images.shape[0]
    
    #1 as a target pose, rest as an input image
    
    batch_vert3d =np.zeros((n_image,t_vert.shape[0],3),np.float32)
    batch_vertuv = np.zeros((n_image,t_vert.shape[0],2),np.float32)
    batch_faces = np.zeros((n_image,t_faces.shape[0],3),np.int32)
    batch_mask = np.zeros((n_image,im_h,im_w,1))
    batch_poses = np.zeros((n_image,1,4,4),np.float32)            
    batch_img =np.zeros((n_image,im_h,im_w,3),np.float32) #input image / output target
    batch_img_gt =np.zeros((n_image,im_h,im_w,3),np.float32) #input image / output target      
    batch_ang_in =np.zeros((n_image,im_h,im_w,1),np.float32) #input image / output target      
    batch_vert3d[:]=t_vert
    batch_faces[:]=t_faces
    
    batch_bbox_train =np.zeros((n_image,4),np.float32) #input image / output target      
    batch_pose_input =np.zeros((n_image,7),np.float32) #input image / output target      
    batch_vert_visible =np.zeros((n_image,t_vert.shape[0],1),np.float32) #input image / output target      


    target_ids=np.arange(n_image)
    #np.random.shuffle(target_ids)
    for b_id,im_id in enumerate(target_ids):
        img_processed = preprocess_input(np.copy(t_images[im_id]) *255 )    
        batch_img[b_id] = img_processed                        
        batch_mask[b_id] = t_masks[im_id]
        batch_vertuv[b_id]=t_vert_uv[im_id] #UV values per vertex
        batch_ang_in[b_id,:,:,0]=t_face_angles[im_id]

        batch_pose_input[b_id]=self.vectorpose(t_poses[im_id])        
        batch_vert_visible[b_id,:,0]=(np.sum(t_vert_uv[im_id],axis=1)>0).astype(np.float32)
        batch_bbox_train[b_id]=t_bboxes_ori[im_id]


    t_id = target_ids[0]
    batch_img_gt[0] = t_images[t_id]
    batch_poses[:,0,:,:] = t_poses[t_id]    
    return batch_img,batch_ang_in,batch_img_gt,batch_vert3d,batch_vertuv,\
           batch_faces,batch_poses,

def get_original_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[..., i]= img[..., i] *std[i]+ mean[i]
    return np.clip(img,0.,1.)


class data_generator():
    def __init__(self,data_dir, back_dir,batch_size=5,target_size=5,**kwargs):                
        '''
        data_dir: Folder that contains cropped image+xyz
        back_dir: Folder that contains random background images
            batch_size: batch size for training
        gan: if False, gt for GAN is not yielded
        '''
        self.data_dir = data_dir
        self.back_dir = back_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.backfiles = os.listdir(back_dir)
        self.datafiles=os.listdir(data_dir)       
        self.n_data = len(self.datafiles)
        self.n_back = len(self.backfiles)  
        
        self.seq_syn= iaa.Sequential([
                                iaa.WithChannels(0, iaa.Add((-15, 15))),
                                iaa.WithChannels(1, iaa.Add((-15, 15))),
                                iaa.WithChannels(2, iaa.Add((-15, 15))),
                                iaa.ContrastNormalization((0.8, 1.3)),
                                iaa.Multiply((0.8, 1.2),per_channel=0.5),
                                iaa.GaussianBlur(sigma=(0.0, 0.5)),
                                iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
                                iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
                                ], random_order=True)
        self.scene_seq = np.arange(self.n_data)
        self.back_seq = np.arange(self.n_back)
        np.random.shuffle(self.scene_seq)
        self.idx=0        
        self.back_idx=0
    def vectorpose(self,mat,clip_delta=False):
        '''
        4x4 matrix to 7 vector (x,y,z,qx,qy,qz,qw)
        '''
        output = np.zeros((6))
        output[:3] = mat[:3,3]
        quat = tf3d.euler.mat2euler(mat[:3,:3]) #tf3d.quaternions.mat2quat(mat[:3,:3]) #w,x,y,z
        output[3:]=quat
        if(clip_delta):
            output[:3] = np.clip(output[:3] ,-0.09,0.09)
            output[3:]=np.clip(output[3:],-0.049,0.049)#rx,ry,rz
        
        #output[6]=quat[0] #qw
        #[x[:,:3]*0.01,x[:,3:]*0.05]
        return output

    def get_a_scene(self,v_input=-1,aug=True,batch_idx=1):
        batch_dummy = np.zeros((self.batch_size))
        batch_latent = np.zeros((self.batch_size,32),np.float32)    
        v_id = self.scene_seq[self.idx]

        self.idx+=1
        if(self.idx>=self.n_data):
            self.idx=0
            np.random.shuffle(self.scene_seq)

        if(v_input>0):
            v_id =v_input
        data_fn = os.path.join(self.data_dir,"{:06d}.hdf5".format(v_id))
        train_data = h5py.File(data_fn, "r")
        t_vert = np.array(train_data["vertices_3d"])
        t_vert_uv = np.array(train_data["vertices_uv"])
        t_faces =  np.array(train_data['faces'])
        t_in_images =  np.array(train_data['in_images'])
        t_gt_images =  np.array(train_data['gt_images'])        
        t_poses_pert = np.array(train_data['poses_pert'])
        t_poses = np.array(train_data['poses'])
        t_face_angles = np.array(train_data['face_angles'])
        #t_bboxes = np.array(train_data['bboxes']).astype(np.float32)

        train_data.close()  

        img_ids = np.arange(t_in_images.shape[0])
        np.random.shuffle(img_ids)
        
        img_id_input = img_ids[:self.batch_size]
        tar_id = min(self.target_size+self.batch_size,img_ids.shape[0])
        img_id_target = img_ids[self.batch_size:tar_id]
        
        
        im_h = t_in_images.shape[1]
        im_w = t_in_images.shape[2]
        #image ddata
        batch_img_in =np.zeros((self.batch_size,im_h,im_w,3),np.float32) #input image / output target
        batch_ang_in =np.zeros((self.batch_size,im_h,im_w,1),np.float32) #input image / output target
        batch_img_gt =np.zeros((self.batch_size,im_h,im_w,3),np.float32) #input image / output target
        
        #mesh data
        batch_vert3d =np.zeros((self.batch_size,t_vert.shape[0],3),np.float32)
        batch_vertuv = np.zeros((self.batch_size,t_vert.shape[0],2),np.float32)
        batch_faces = np.zeros((self.batch_size,t_faces.shape[0],3),np.int32)
        #pose data
        batch_poses = np.zeros((self.batch_size,self.target_size,4,4),np.float32)                    
        #rendering info

        batch_bbox_train =np.zeros((self.batch_size,4),np.float32) #input image / output target      
        batch_bbox_train[:,2]=255
        batch_bbox_train[:,3]=255
        
        batch_vert3d[:]=t_vert
        batch_faces[:]=t_faces   
        
        
        batch_pose_input =np.zeros((self.batch_size,4,4),np.float32) #input image / output target      
        batch_pose_delta_gt =np.zeros((self.batch_size,6),np.float32) #input image / output target      
        batch_vert_visible =np.zeros((self.batch_size,t_vert.shape[0],1),np.float32) #input image / output target      
        
        
        #batch_img_in,batch_img_gt,batch_vert3d,batch_vertuv,batch_faces,batch_poses
        for b_id, img_id in enumerate(img_id_input):
            img_in = np.copy(t_in_images[img_id])
            if(np.max(img_in)>2):
                    img_aug = img_in
            else:
                    img_aug = img_in*255
            if(aug):
                img_aug = self.seq_syn.augment_image(img_aug) 

            img_aug = preprocess_input(img_aug)
            #img_aug = img_aug/255.
            batch_img_in[b_id] = img_aug 
            batch_vertuv[b_id] = t_vert_uv[img_id]
            batch_ang_in[b_id,:,:,0] = t_face_angles[img_id]
            pose_noise = t_poses_pert[img_id] #noisy pose 4x4
            pose_ori = t_poses[img_id] #correct pose 4x4
            delta_pose = np.matmul(pose_ori,np.linalg.inv(pose_noise))
            
            batch_pose_input[b_id]=pose_noise#self.vectorpose(pose_noise)
            batch_pose_delta_gt[b_id]=self.vectorpose(delta_pose,clip_delta=True) #np.array([0,0,0,0,0,0,1])
            
            batch_vert_visible[b_id,:,0]=(np.sum(t_vert_uv[img_id],axis=1)>0).astype(np.float32)
        
        ids_inputs =img_ids[:self.batch_size]
        np.random.shuffle(ids_inputs)
        if(batch_idx==-1):
            img_id_target=ids_inputs
            #put seen pose to force network preserve the images from the same pose
            
        for b_id, img_id in enumerate(img_id_target[:self.target_size]):
            img_in = np.copy(t_gt_images[img_id])
            if(np.max(img_in)>2):
                img_aug = preprocess_input(img_in) #no augmetnation for target
            else:
                img_aug = preprocess_input(img_in*255)   #no augmetnation for target         
            #img_aug = img_aug/255.
            batch_img_gt[b_id] = img_aug                
            batch_poses[:,b_id,:] = np.expand_dims(t_poses[img_id],axis=0)
        
        return batch_img_in,batch_ang_in,batch_img_gt,batch_vert3d,\
               batch_vertuv,batch_faces,batch_poses,\
               batch_pose_input,batch_pose_delta_gt,batch_bbox_train,\
               batch_vert_visible

    def generator(self):   
        batch_idx=1     
        while True:
            batch_idx=-batch_idx
            #img_in,ang_in,img_gt,vert3d,vertuv,faces,poses = self.get_a_scene(batch_idx=batch_idx)
            img_in,ang_in,img_gt,vert3d,vertuv,faces,poses,\
            pose_input,pose_delta_gt,bbox_train,vert_visible=\
            self.get_a_scene(batch_idx=batch_idx)           
            if(vert3d.shape[1]>20000):
                continue
            yield [img_in,ang_in,vert3d,faces,
                   pose_input,vert_visible,bbox_train,
                   poses,img_gt,pose_delta_gt],\
                  [np.zeros((self.batch_size))]
        
    def generator_old(self):        
        idx=0
        back_idx=0
        

        while True:
            v_id = scene_seq[idx]
            idx+=1
            if(idx>=self.n_data):
                idx=0
                np.random.shuffle(scene_seq)
            
            data_fn = os.path.join(self.data_dir,"{:06d}.hdf5".format(v_id))
            train_data = h5py.File(data_fn, "r")
            t_vert = np.array(train_data["vertices_3d"])
            t_vert_uv = np.array(train_data["vertices_uv"])
            t_faces =  np.array(train_data['faces'])
            t_images =  np.array(train_data['images'])
            t_gt_images =  np.array(train_data['gt_images'])
            
            t_uvs = np.array(train_data['uvs'])
            t_poses = np.array(train_data['poses'])
            t_texture = np.array(train_data['texture'])
            t_light = np.array(train_data['lights'])
            train_data.close()  

            img_ids = np.arange(10)
            np.random.shuffle(img_ids)
            
            im_h = t_images.shape[1]
            im_w = t_images.shape[2]
            #image ddata
            batch_img =np.zeros((self.batch_size,im_h,im_w,3),np.float32) #input image / output target
            batch_img_gt =np.zeros((self.batch_size,im_h,im_w,3),np.float32) #input image / output target
            batch_img_uv =np.zeros((self.batch_size,im_h,im_w,2),np.float32) #input image / output target
            batch_mask = np.zeros((self.batch_size,im_h,im_w,1))
            #mesh data
            batch_vert3d =np.zeros((self.batch_size,t_vert.shape[0],3),np.float32)
            batch_vertuv = np.zeros((self.batch_size,t_vert_uv.shape[0],2),np.float32)
            batch_faces = np.zeros((self.batch_size,t_faces.shape[0],3),np.int32)
            #pose data
            batch_poses = np.zeros((self.batch_size,4,4),np.float32)            
            batch_textures = np.zeros((self.batch_size,128,128,3),np.float32)            
            #rendering info
            batch_lights = np.zeros((self.batch_size,9),np.float32)          


            img_ids = img_ids[:self.batch_size]
            batch_vert3d[:]=t_vert
            batch_vertuv[:]=t_vert_uv
            batch_faces[:]=t_faces
            batch_textures[:]=t_texture
            for b_id, img_id in enumerate(img_ids):
                #occ_mask 
                
                #define a area -> remove uv --> mask --> filed with background
                occ_w = int(random.random()*128)
                occ_h = int(random.random()*128)
                pos_v = int(random.random()*256)-128
                pos_h = int(random.random()*256)-128
                occ_area = [pos_v+128,pos_h+128,pos_v+128+occ_h,pos_h+128+occ_w]
                occ_area = np.clip(occ_area,0,256)
                t_uv = t_uvs[img_id]
                t_uv[occ_area[0]:occ_area[2],occ_area[1]:occ_area[3]]=0                
                t_mask = np.sum(t_uv,axis=2,keepdims=True)>0                
                if(np.sum(t_mask)<10):
                    t_uv = t_uvs[img_id]
                    t_mask = np.sum(t_uv,axis=2,keepdims=True)>0                
                    #no occlusion 
                #dilation or erode masks

                #Augmentation of backgrounds image / occlusion
                img_in = np.copy(t_images[img_id])
                if(np.max(img_in)>100):
                    img_in =img_in/255.0
                    
                mask_img = np.sum(img_in,axis=2,keepdims=True)>0
                back_fn = os.path.join(self.back_dir,self.backfiles[back_seq[back_idx]])
                back_img = cv2.imread(back_fn)[:,:,::-1]/255.0
                back_img = resize(back_img,(256+10,256+10))
                
                shift_v = 0#random.randint(-5,5)
                shift_u = 0#random.randint(-5,5)
                
                back_img[5+shift_v:5+shift_v+256,5+shift_u:5+256+shift_u]= \
                mask_img*img_in+(1-mask_img)*back_img[5+shift_v:5+shift_v+256,5+shift_u:5+256+shift_u]
                
                '''
                mask_change = random.randint(-5,5)
                shifted_mask = np.copy(t_mask)
                if(mask_change<0):
                    for i in range(-mask_change):
                        shifted_mask=morphology.binary_erosion(shifted_mask)
                else:
                    for i in range(mask_change):
                        shifted_mask=morphology.binary_dilation(shifted_mask) 
                ''' 
                img_aug = np.clip(back_img[5:5+256,5:5+256],0,1)*255
                img_aug = self.seq_syn.augment_image(img_aug)                                        
                img_aug[np.invert(t_mask[:,:,0])]=0                
                img_aug = preprocess_input(img_aug)
                #img_aug = img_aug/255.
                
                #remove background using augmented mask                
                #zero centering of color values #0~1 -> #-1~1
                t_pose = t_poses[img_id]      

                batch_poses[b_id] = t_pose
                batch_img[b_id] = img_aug                
                batch_img_uv[b_id] =t_uv
                if(np.max(t_gt_images[img_id])>100):                    
                    batch_img_gt[b_id] = preprocess_input(t_gt_images[img_id])
                else:
                    batch_img_gt[b_id] = preprocess_input(t_gt_images[img_id]*255)

                batch_mask[b_id] = t_mask
                #batch_lights[b_id]=t_light[img_id]
                back_idx+=1
                if(back_idx >= self.n_back)  :
                    np.random.shuffle(back_seq)
                    back_idx=0
            
            yield [batch_img,batch_img_uv,batch_mask,
                   batch_vert3d,batch_vertuv,batch_faces,
                   batch_poses,batch_img_gt,batch_textures,batch_lights],\
                  [batch_dummy,batch_dummy,batch_dummy]
            #yield [batch_img,batch_img_uv,batch_mask],[batch_dummy]#[batch_img]
            '''
            yield [batch_img,batch_img_uv,batch_mask,
                   batch_vert3d,batch_vertuv,batch_faces,
                   batch_poses,batch_latent,batch_textures,batch_img_gt],\
                  [batch_dummy,batch_dummy,batch_dummy]
            '''
