import tensorflow as tf
import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting

import sys,os
import numpy as np
from tensorflow.keras.layers import Layer

def build_projection(cam,  w=640, h=480, x0=0, y0=0, nc=0.1, fc=10.0):
    q = -(fc + nc) / float(fc - nc)
    qn = -2 * (fc * nc) / float(fc - nc)
    proj = np.array([
        [2 * cam[0, 0] / w, -2 * cam[0, 1] / w, (-2 * cam[0, 2] + w + 2 * x0) / w, 0],
        [0, -2 * cam[1, 1] / h, (-2 * cam[1, 2] + h + 2 * y0) / h, 0],
        [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes
        [0, 0, -1, 0]
    ])        
    proj[1, :] *= -1.0
    return proj.T

class neural_rendering_crop_resize(Layer):
    def __init__(self,img_h,img_w,cam_K,target_h=-1,target_w=-1,near=0.1,far=10.0,ch_dim=3,**kwargs):
        self.img_h=img_h
        self.img_w=img_w
        if(target_h==-1):
            self.target_h=img_h
            self.target_w=img_w            
        else:
            self.target_h=target_h
            self.target_w=target_w            
        self.cam_K = cam_K
        self.near = near
        self.far = far        
        #self.right = ((self.img_w-1.)/2.)*near/self.focal
        self.projection_matrix = tf.constant(build_projection(cam_K,w=self.img_w,h=self.img_h),tf.float32)        
        self.ch_dim=ch_dim
        super(neural_rendering_crop_resize,self).__init__(**kwargs)
    def build(self,input_shape):
        super(neural_rendering_crop_resize,self).build(input_shape)
    def call(self,x):
        #N: num. of vertices, F: num of faces
        vertices = x[0] #(batchxNx3) #same for each batch
        uv_map = x[1] #(batchxNx2) #different for each batch
        faces = tf.cast(x[2],tf.int32) #batchxFx3, same for each batch
        texture = x[3] #batchxWxHxCH different for each batch
        poses=x[4] #batch x n_target_poses x 4x4, same for each batch
        bboxes=x[5] #batch x n_target_poses x 4 -> should be normalized by the full res
        #ignore batch dimension of poses
        vertices_mult = tf.tile(vertices,[tf.shape(poses)[1],1,1])
        vert_uvs_mult = tf.tile(uv_map,[tf.shape(poses)[1],1,1])
        faces_multi = tf.tile(faces,[tf.shape(poses)[1],1,1])
        texture_multi = tf.tile(texture,[tf.shape(poses)[1],1,1,1])        
        poses_t=tf.transpose(poses,[1,0,2,3]) #posexbatchx4x4
        poses_t=tf.reshape(poses_t,[-1,4,4])
        bboxes_t = tf.transpose(bboxes,[1,0,2]) 
        bboxes_t=tf.reshape(bboxes_t,[-1,4]) #(posexbatch)x4
        
        # Transform vertices from camera to clip space        
        vertices_objects, vertices_cameras,vertices_clips,vertices_normals,view_matrices=\
           tf.map_fn(self.transform_vertices,(vertices_mult,poses_t,faces_multi),dtype=(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32))        

        gbuffer_temp = dirt.rasterise_batch(
                        background=tf.zeros([tf.shape(vertices_mult)[0],self.img_h, self.img_w, 3]),
                        vertices=vertices_clips,
                        vertex_colors=tf.concat([
                            tf.ones_like(vertices_objects[:,:, :1]),  #1 mask
                            vert_uvs_mult                            
                        ], axis=2),
                        faces=faces_multi,                                                
                        height=self.img_h,
                        width=self.img_w,
                        channels=3
                       )    
        rendered_feature_raw = tf.map_fn(self.sample_texture,(texture_multi, gbuffer_temp[:,:,:,1:3] ),dtype=tf.float32)        
        #if both uv value is zero -> 
        uv_projection = gbuffer_temp[:,:,:,1:3]
        mask_old = gbuffer_temp[:,:,:,:1] #regardless of the facts that each pixel was seen by the input images
        
        if not(self.target_h==self.img_h and self.target_h==self.img_h):
            #for the same pose -> same crop and resize area                        
            mask_old = tf.image.crop_and_resize(mask_old,bboxes_t, 
                                                 crop_size=(self.target_h,self.target_w),
                                                 box_indices=tf.range(0,tf.shape(rendered_feature_raw)[0]))
            mask_old = tf.cast(tf.greater(mask_old,0.5),tf.float32)
            mask_rend = tf.cast(tf.greater(tf.reduce_sum(gbuffer_temp[:,:,:,1:3],axis=3,keepdims=True),0),tf.float32)            
            mask_crop = tf.image.crop_and_resize(mask_rend,bboxes_t, 
                                                 crop_size=(self.target_h,self.target_w),
                                                 box_indices=tf.range(0,tf.shape(rendered_feature_raw)[0]))
            mask_new = tf.cast(tf.greater(mask_crop,0.5),tf.float32)            
            rendered_feature = tf.image.crop_and_resize(rendered_feature_raw ,bboxes_t, 
                               crop_size=(self.target_h,self.target_w),
                               box_indices=tf.range(0,tf.shape(rendered_feature_raw)[0]))
            
            uv_projection = tf.image.crop_and_resize(uv_projection ,bboxes_t, 
                               crop_size=(self.target_h,self.target_w),
                               box_indices=tf.range(0,tf.shape(rendered_feature_raw)[0]))                              
        else:
            mask_new = tf.cast(tf.greater(tf.reduce_sum(gbuffer_temp[:,:,:,1:3],axis=3,keepdims=True),0),tf.float32)            
            rendered_feature = mask_new*rendered_feature_raw #remove backgrounds
        
        concated_out = tf.concat([mask_new,rendered_feature,mask_old,uv_projection],axis=3) # P X B x H x W x CH        
        final_out = tf.reshape(concated_out, [tf.shape(poses)[1],-1,self.target_h,self.target_w,self.ch_dim+4])
        #(batch*n_poses) x H x W x (ch+1) -> (n_poses x batch x H x W x (ch+1))
        #pack each image in a pose
        return final_out  

    def transform_vertices(self,inputs):
        vertices = inputs[0]
        pose = inputs[1]
        faces= inputs[2]
        cube_vertices_object = tf.concat([
            vertices,
            tf.ones_like(vertices[:, -1:])
        ], axis=1)
        cube_normals_world = lighting.vertex_normals_pre_split(cube_vertices_object, faces)
        transform_gl = tf.constant([[1,0,0],[0,-1,0],[0,0,-1]],tf.float32)
        tensor_rot = tf.matmul(transform_gl,pose[:3,:3])
        rot_list = tf.unstack(tf.reshape(tensor_rot,[-1]))    
        pose_list = tf.unstack(tf.reshape(pose,[-1]))
        pose_list[0:3]=rot_list[0:3]
        pose_list[4:7]=rot_list[3:6]
        pose_list[8:11]=rot_list[6:9]
        pose_list[7]=-pose_list[7]
        pose_list[11]=-pose_list[11]           
        cam_pose = tf.stack(pose_list)
        cam_pose = tf.reshape(cam_pose,(4,4))
        view_matrix = tf.transpose(cam_pose)
        cube_vertices_camera = tf.matmul(cube_vertices_object, view_matrix)
        cube_vertices_clip = tf.matmul(cube_vertices_camera, self.projection_matrix)
        return cube_vertices_object,cube_vertices_camera,cube_vertices_clip,cube_normals_world,view_matrix
    def uvs_to_pixel_indices(self,uvs, texture_shape):
        # Note that this assumes u = 0, v = 0 is at the top-left of the image -- different to OpenGL!
        uvs = uvs[..., ::-1]  # change x, y coordinates to y, x indices
        #batchxhxwx2  vs [2]
        texture_shape = tf.cast(texture_shape, tf.float32) #[h,w]
        return tf.clip_by_value(uvs, 0., 1.) * texture_shape        

    def sample_texture(self,inputs):
        texture=inputs[0]
        uv_val_ori=inputs[1] #wxhx2
        indices = self.uvs_to_pixel_indices(uv_val_ori, tf.shape(texture)[:2])

        floor_indices = tf.floor(indices)
        frac_indices = indices - floor_indices
        floor_indices = tf.cast(floor_indices, tf.int32)
        neighbours = tf.gather_nd(
            texture,
            tf.stack([
                floor_indices,
                floor_indices + [0, 1],
                floor_indices + [1, 0],
                floor_indices + [1, 1]
            ]),
        )
        top_left, top_right, bottom_left, bottom_right = tf.unstack(neighbours)
        return \
            top_left * (1. - frac_indices[..., 1:]) * (1. - frac_indices[..., :1]) + \
            top_right * frac_indices[..., 1:] * (1. - frac_indices[..., :1]) + \
            bottom_left * (1. - frac_indices[..., 1:]) * frac_indices[..., :1] + \
            bottom_right * frac_indices[..., 1:] * frac_indices[..., :1]

    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[4][1],input_shape[0][0],self.target_h,self.target_w,self.ch_dim+4]))

class neural_rendering_gbuffer(Layer):
    def __init__(self,img_h,img_w,cam_K,near=0.1,far=10.0,ch_dim=3,**kwargs):
        self.img_h=img_h
        self.img_w=img_w       
        self.cam_K = cam_K
        self.near = near
        self.far = far        
        #self.right = ((self.img_w-1.)/2.)*near/self.focal
        self.projection_matrix = tf.constant(build_projection(cam_K,w=self.img_w,h=self.img_h),tf.float32)        
        super(neural_rendering_gbuffer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(neural_rendering_gbuffer,self).build(input_shape)
    def call(self,x):
        #N: num. of vertices, F: num of faces
        vertices = x[0] #(1xNx3) #same for each batch
        faces = tf.cast(x[1],tf.int32) #1xFx3, same for each batch
        poses=x[2] #1 x 4x4, same for each batch        
        
        # Transform vertices from camera to clip space        
        vert_obj,vert_3d,vert_clip,normals= self.transform_vertices(vertices[0],poses[0],faces[0])

        gbuffer_temp = dirt.rasterise(
                        background=tf.zeros([self.img_h, self.img_w, 11]),
                        vertices=vert_clip,
                        vertex_colors=tf.concat([
                            tf.ones_like(vert_obj[:, :1]),  #1 mask
                            vert_3d,
                            normals,
                            vert_obj                        
                        ], axis=1),
                        faces=faces[0],                                                
                        height=self.img_h,
                        width=self.img_w,
                        channels=11
                       )    
        return tf.expand_dims(gbuffer_temp,axis=0)

    def transform_vertices(self,vertices,pose,faces):
        #vertices = inputs[0]
        #pose = inputs[1]
        #faces= inputs[2]
        cube_vertices_object = tf.concat([
            vertices,
            tf.ones_like(vertices[:, -1:])
        ], axis=1)
        cube_normals_world = lighting.vertex_normals_pre_split(cube_vertices_object, faces)
        transform_gl = tf.constant([[1,0,0],[0,-1,0],[0,0,-1]],tf.float32)
        tensor_rot = tf.matmul(transform_gl,pose[:3,:3])
        rot_list = tf.unstack(tf.reshape(tensor_rot,[-1]))    
        pose_list = tf.unstack(tf.reshape(pose,[-1]))
        pose_list[0:3]=rot_list[0:3]
        pose_list[4:7]=rot_list[3:6]
        pose_list[8:11]=rot_list[6:9]
        pose_list[7]=-pose_list[7]
        pose_list[11]=-pose_list[11]           
        cam_pose = tf.stack(pose_list)
        cam_pose = tf.reshape(cam_pose,(4,4))
        view_matrix = tf.transpose(cam_pose)
        cube_vertices_camera = tf.matmul(cube_vertices_object, view_matrix)
        cube_vertices_3d = tf.transpose(tf.matmul(pose[:3,:3], tf.transpose(cube_vertices_object[:,:3])))+tf.transpose(pose[:3,3:4]) #3xN

        cube_vertices_clip = tf.matmul(cube_vertices_camera, self.projection_matrix)
        return cube_vertices_object,cube_vertices_3d,cube_vertices_clip,cube_normals_world
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.img_h,self.img_w,7]))