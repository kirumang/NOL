import tensorflow as tf
import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting

import sys,os
from rendering.model import Model3D
import numpy as np
from keras.layers import Layer


class proj_to_tex_layer(Layer):
    '''
    Project feature map into the texture feature space using proposed weighted average.
    arg:tex_h,tex_w,f_ch #target texture res and channel, f_ch==channel of input feature map
    [input]
    uv_map = Batch x W x H x 2
    feature_map = Batch x W x H x f_ch
    mask = Batch x W x H x 1    
    [output]
    - a projected_texture_map after Weighted average over batch.
      Batch x tex_h x tex_w x f_ch
      Thus, every batch has the same value
      -> output[0,x,y,z]==output[1,x,y,z]==output[n,x,y,z] for any x,y,and z    
    '''
    def __init__(self,tex_h,tex_w,f_ch,d_factor=0.5,**kwargs):
        self.tex_h=tex_h
        self.tex_w=tex_w
        self.f_ch=f_ch
        self.decay_factor=d_factor #should proportional to the tex_h,tex_w
        self.decay = np.sqrt(self.decay_factor**2+self.decay_factor**2)
        self.active_field=int(-np.log(0.1)*self.decay+ 0.5) #round up

        super(proj_to_tex_layer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(proj_to_tex_layer,self).build(input_shape)
    def call(self,x):
        f_map = x[0] #w,h,f_ch
        uv_map = x[1] #w,h,2
        mask = x[2]   #w,h,1
        pack_inputs =tf.concat([mask,uv_map,f_map],axis=3) #Batch x W x H, chs
        textures = tf.map_fn(self.project_to_texture,pack_inputs)        
        weighted_sum =tf.reduce_sum(textures[:,:,:,1:],axis=0,keep_dims=True) 
        weight_sum  = tf.reduce_sum(textures[:,:,:,:1],axis=0,keep_dims=True)
        weighted_sum = tf.cast(tf.greater(weight_sum,0.01),tf.float32)*weighted_sum
        unified_tex = weighted_sum / (weight_sum+0.001)        
        return tf.tile(unified_tex,[tf.shape(f_map)[0],1,1,1])
    
    def project_to_texture(self,packed_input):
        '''
        W,H -> size of an input image or a feature map
        input
        - uvs_map: rendered UV coordinates (Tensor: HxWx2 -> v/u order, tf.float32)
        - pixels: rendered(or real) RGB values (Tensor: HxWx3 -> r,g,b order, tf.float32)
        - masks: object_mask (Tensor: WxHx1 -> 1:object, 0:background, boolean)    
        - (opt) tex_h: height of the texture map (of the neural texture)
        - (opt) tex_w: width of the texture map (of the neural texture)
        - (opt) f_ch: number of channels of the texturemap (e.g.,3 for rgb textures)
        
        output
        - w_map: sum of accumulated weights of each pixels (Tensor: tex_h x tex_w x 1)
        - w_color: sum of accumulated weighted colors(or features) of each pixels (Tensor: tex_h x tex_w x ch, ch=3 for rgb images)
        e.g., final weighted average can be calculated by: norm_tex = w_color/(w_map+0.0001)
        '''
        #uvs_map,pixels,masks
        masks = packed_input[:,:,0]
        uvs_map = packed_input[:,:,1:3]
        pixels = packed_input[:,:,3:]
        tex_size= tf.constant([self.tex_h,self.tex_w],tf.float32)
        mask_indices = tf.where(tf.equal(masks,1))
        uvs_map = uvs_map[..., ::-1] 
        uvs_map = tf.multiply(uvs_map,tex_size)
        uv_gathered = tf.gather_nd(uvs_map,mask_indices)
        pix_gathered = tf.gather_nd(pixels,mask_indices)        
        
        weighted_maps = tf.zeros([self.tex_h, self.tex_w, 1+self.f_ch],tf.float32) #tf.constant(np.zeros((self.tex_h, self.tex_w, 1+self.f_ch)),tf.float32) #0:weight, 1,2,3:weighted_sum (r,g,b)         
        
        v_shift = tf.expand_dims(tf.maximum(uv_gathered[:,0]-self.active_field,0),axis=1) #Nx1
        u_shift = tf.expand_dims(tf.maximum(uv_gathered[:,1]-self.active_field,0),axis=1) #Nx1
        v_max = tf.expand_dims(tf.minimum(uv_gathered[:,0]+self.active_field,self.tex_h-1),axis=0)
        u_max = tf.expand_dims(tf.minimum(uv_gathered[:,1]+self.active_field,self.tex_w-1),axis=0)     
        u_grid,v_grid = tf.meshgrid(tf.range(0,2*self.active_field),tf.range(0,2*self.active_field))
        v_grid = tf.tile(tf.expand_dims(tf.cast(v_grid,tf.float32),axis=0),[tf.shape(v_shift)[0],1,1]) + tf.expand_dims(v_shift,axis=2) #Nx4x4 + NX1
        u_grid = tf.tile(tf.expand_dims(tf.cast(u_grid,tf.float32),axis=0),[tf.shape(u_shift)[0],1,1]) + tf.expand_dims(u_shift,axis=2) #Nx4x4
        tex_u = tf.expand_dims(tf.expand_dims(uv_gathered[:,1],axis=1),axis=2) #Nx1
        tex_v = tf.expand_dims(tf.expand_dims(uv_gathered[:,0],axis=1),axis=2)#Nx1
        
        diff_u = tf.minimum(tf.abs(u_grid -tex_u ),tf.abs(tf.abs(u_grid - tex_u)-self.tex_w))
        diff_v = tf.minimum(tf.abs(v_grid -tex_v),tf.abs(tf.abs(v_grid - tex_v)-self.tex_h))                        
        dist_map = tf.sqrt(tf.square(diff_u)+tf.square(diff_v))
        weight_map = tf.exp(-dist_map/self.decay) #NX4x4                            
        weight_fin =tf.cast(tf.greater(weight_map,0.1),tf.float32)*weight_map#Nx4x4
        weighted_sum =  tf.expand_dims(weight_fin,axis=3)* tf.expand_dims(tf.expand_dims(pix_gathered,axis=1),axis=1)  #Nx4x4x1 * Nx1x1x3
        #v_grid[Nx4x4],u_grid[Nx4x4],weight_fin[Nx4x4], weighted_sum[Nx4x4x3]
        indices = tf.cast(tf.stack([tf.reshape(v_grid,[-1]),tf.reshape(u_grid,[-1])],axis=1),tf.int64)        
        #Nx2
        values_w = tf.reshape(weight_fin,[-1,1])
        values_w_sum = tf.reshape(weighted_sum,[-1,self.f_ch])
        values_concat = tf.concat([values_w,values_w_sum],axis=1) #Nx(1+f_ch)
        output = tf.tensor_scatter_nd_add(weighted_maps,indices,values_concat) 
        return output
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.tex_h,self.tex_w,self.f_ch]))

class proj_to_tex_layer_rnn(proj_to_tex_layer):
    '''
    Project feature map into the texture feature space using proposed weighted average.
    arg:tex_h,tex_w,f_ch #target texture res and channel, f_ch==channel of input feature map
    [input]
    uv_map = Batch x W x H x 2
    feature_map = Batch x W x H x f_ch
    mask = Batch x W x H x 1    
    [output]
    - a projected_texture_map after Weighted average over batch.
      Batch x tex_h x tex_w x f_ch
      Thus, every batch has the same value
      -> output[0,x,y,z]==output[1,x,y,z]==output[n,x,y,z] for any x,y,and z    
    '''
    def __init__(self,**kwargs):
        #self.tex_h=tex_h
        #self.tex_w=tex_w
        #self.f_ch=f_ch
        #self.decay_factor=d_factor #should proportional to the tex_h,tex_w
        #self.decay = np.sqrt(self.decay_factor**2+self.decay_factor**2)
        #self.active_field=int(-np.log(0.1)*self.decay+ 0.5) #round up
        super(proj_to_tex_layer_rnn,self).__init__(**kwargs)
    def build(self,input_shape):
        super(proj_to_tex_layer_rnn,self).build(input_shape)
    def call(self,x):
        f_map = x[0] #w,h,f_ch
        uv_map = x[1] #w,h,2
        mask = x[2]   #w,h,1
        pack_inputs =tf.concat([mask,uv_map,f_map],axis=3) #Batch x W x H, chs
        textures = tf.map_fn(self.project_to_texture,pack_inputs)  
        batch_avg =tf.reduce_sum(textures,axis=0,keep_dims=True)        
        output = tf.concat([batch_avg,textures],axis=0)
        #output -> 1 x(1:avg +batch) x w x h x 4 (weight/weighted_sum)
        return output #converted to the LSTM input
        
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.tex_h,self.tex_w,self.f_ch+1]))


class neural_rendering_raster_and_texture(Layer):
    def __init__(self,img_h,img_w,focal,near=0.1,far=10.0,**kwargs):
        self.img_h=img_h
        self.img_w=img_w
        self.focal = focal
        self.near = near
        self.far = far        
        self.right = ((self.img_w-1.)/2.)*near/self.focal
        self.projection_matrix = matrices.perspective_projection(near=self.near, far=self.far, right=self.right, aspect=float(self.img_h) / self.img_w)    
        super(neural_rendering_raster_and_texture,self).__init__(**kwargs)
    def build(self,input_shape):
        super(neural_rendering_raster_and_texture,self).build(input_shape)
    def call(self,x):
        #N: num. of vertices, F: num of faces
        vertices = x[0] #(batchxNx3)
        uv_map = x[1] #(batchxNx2)
        faces = tf.cast(x[2],tf.int32) #Fx3
        texture = x[3]
        poses=x[4] #batchx4x4
        #inputs = (vertices_xyz,vertices_uv,faces,textures,poses)
        #vertices_xyz,vertices_uv,faces,textures,poses
        '''
        vertices: vertices (N,3) , tf.float32
        uv_map: uv_map (N,2), (normal GL convention) , tf.float32
        faces : faces (N,3) , tf.int32
        texture = (WxHx3 or C)  (0~1) texture image, tf.float32
        pose : (4x4)
        '''        
        # Transform vertices from camera to clip space        
        vertices_objects, vertices_cameras,vertices_clips=\
           tf.map_fn(self.transform_vertices,(vertices,poses),dtype=(tf.float32,tf.float32,tf.float32))        

        mask_uv = dirt.rasterise_batch(
                        background=tf.zeros([tf.shape(x[3])[0],self.img_h, self.img_w, 3]),
                        vertices=vertices_clips,
                        vertex_colors=tf.concat([
                            tf.ones_like(vertices_objects[:,:, :1]),  # mask
                            uv_map  # texture coordinates #cub e_normals_world,  # normals
                                    #cube_vertices_cameras[:,:, :3]
                        ], axis=2),
                        faces=faces,                                                
                        height=self.img_h,
                        width=self.img_w,
                        channels=3
                       )                
        
        mask_render = tf.expand_dims(mask_uv[:,:, :, 0],axis=3)
        unlit_colors = tf.map_fn(self.sample_texture,(texture, mask_uv[:,:,:,1:3] ),dtype=tf.float32)        
        #batc x h x w x 3   x batch x h x w x  
        pixels = unlit_colors * mask_render
        
        return pixels #tf.concat([pixels],axis=3)

    def transform_vertices(self,inputs):
        vertices = inputs[0]
        pose = inputs[1]
        cube_vertices_object = tf.concat([
            vertices,
            tf.ones_like(vertices[:, -1:])
        ], axis=1)
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
        return cube_vertices_object,cube_vertices_camera,cube_vertices_clip
    def uvs_to_pixel_indices(self,uvs, texture_shape):
        # Note that this assumes u = 0, v = 0 is at the top-left of the image -- different to OpenGL!
        uvs = uvs[..., ::-1]  # change x, y coordinates to y, x indices
        #batchxhxwx2  vs [2]
        texture_shape = tf.cast(texture_shape, tf.float32) #[h,w]
        return tf.clip_by_value(uvs, 0., 1.) * texture_shape        

    def sample_texture(self,inputs):
        texture=inputs[0]
        uv_val_ori=inputs[1] #0~1
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

    def shader_fn(self,gbuffer, texture):
            mask = gbuffer[:,:, :, :1]
            uvs = gbuffer[:,:, :, 1:3]
            #positions = gbuffer[:,:, :, 3:]
            unlit_colors = tf.map_fn(self.sample_texture,(texture, self.uvs_to_pixel_indices(uvs, tf.shape(texture)[:2]) ),dtype=tf.float32)
            pixels = unlit_colors * mask # + [0.] * (1. - mask)
            return pixels,mask#,positions            
    
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.img_h,self.img_w,4]))

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

class neural_rendering_raster_lighting(Layer):
    def __init__(self,img_h,img_w,cam_K,near=0.1,far=10.0,ch_dim=3,**kwargs):
        self.img_h=img_h
        self.img_w=img_w
        self.cam_K = cam_K
        self.near = near
        self.far = far        
        #self.right = ((self.img_w-1.)/2.)*near/self.focal
        self.projection_matrix = tf.constant(build_projection(cam_K,w=self.img_w,h=self.img_h),tf.float32)        
        self.ch_dim=ch_dim
        super(neural_rendering_raster_lighting,self).__init__(**kwargs)
    def build(self,input_shape):
        super(neural_rendering_raster_lighting,self).build(input_shape)
    def call(self,x):
        #N: num. of vertices, F: num of faces
        vertices = x[0] #(batchxNx3) #same for each batch
        uv_map = x[1] #(batchxNx2) #different for each batch
        faces = tf.cast(x[2],tf.int32) #batchxFx3, same for each batch
        texture = x[3] #batchxWxHxCH different for each batch
        poses=x[4] #batch x n_target_poses x 4x4, same for each batch
        #ignore batch dimension of poses
        vertices_mult = tf.tile(vertices,[tf.shape(poses)[1],1,1])
        vert_uvs_mult = tf.tile(uv_map,[tf.shape(poses)[1],1,1])
        faces_multi = tf.tile(faces,[tf.shape(poses)[1],1,1])
        texture_multi = tf.tile(texture,[tf.shape(poses)[1],1,1,1])        
        poses_t=tf.transpose(poses,[1,0,2,3])
        poses_t=tf.reshape(poses_t,[-1,4,4])

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
        rendered_feature = tf.map_fn(self.sample_texture,(texture_multi, gbuffer_temp[:,:,:,1:3] ),dtype=tf.float32)        
        #if both uv value is zero -> 
        #mask_old = gbuffer_temp[:,:,:,:1] #regardless of the facts that each pixel was seen by the input images
        mask_new = tf.cast(tf.greater(tf.reduce_sum(gbuffer_temp[:,:,:,1:3],axis=3,keepdims=True),0),tf.float32)
        rendered_feature = mask_new*rendered_feature #remove backgrounds
        concated_out = tf.concat([mask_new,rendered_feature],axis=3)
        final_out = tf.reshape(concated_out, [tf.shape(poses)[1],-1,self.img_h,self.img_w,self.ch_dim+1])
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
        return (tuple([input_shape[4][1],input_shape[0][0],self.img_h,self.img_w, self.ch_dim+1]))




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
        #mask_old = gbuffer_temp[:,:,:,:1] #regardless of the facts that each pixel was seen by the input images
        
        
        if not(self.target_h==self.img_h and self.target_h==self.img_h):
            #for the same pose -> same crop and resize area                        
            mask_rend = tf.cast(tf.greater(tf.reduce_sum(gbuffer_temp[:,:,:,1:3],axis=3,keepdims=True),0),tf.float32)            
            mask_crop = tf.image.crop_and_resize(mask_rend,bboxes_t, 
                                                 crop_size=(self.target_h,self.target_w),
                                                 box_indices=tf.range(0,tf.shape(rendered_feature_raw)[0]))
            mask_new = tf.cast(tf.greater(mask_crop,0.5),tf.float32)            
            rendered_feature = tf.image.crop_and_resize(rendered_feature_raw ,bboxes_t, 
                               crop_size=(self.target_h,self.target_w),
                               box_indices=tf.range(0,tf.shape(rendered_feature_raw)[0]))
        else:
            mask_new = tf.cast(tf.greater(tf.reduce_sum(gbuffer_temp[:,:,:,1:3],axis=3,keepdims=True),0),tf.float32)            
            rendered_feature = mask_new*rendered_feature_raw #remove backgrounds
        concated_out = tf.concat([mask_new,rendered_feature],axis=3) # P X B x H x W x CH        
        final_out = tf.reshape(concated_out, [tf.shape(poses)[1],-1,self.target_h,self.target_w,self.ch_dim+1])
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
        return (tuple([input_shape[4][1],input_shape[0][0],self.target_h,self.target_w,self.ch_dim+1]))


        import tensorflow as tf
import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting

import sys,os
from rendering.model import Model3D
import numpy as np
from keras.layers import Layer


class proj_to_tex_layer(Layer):
    '''
    Project feature map into the texture feature space using proposed weighted average.
    arg:tex_h,tex_w,f_ch #target texture res and channel, f_ch==channel of input feature map
    [input]
    uv_map = Batch x W x H x 2
    feature_map = Batch x W x H x f_ch
    mask = Batch x W x H x 1    
    [output]
    - a projected_texture_map after Weighted average over batch.
      Batch x tex_h x tex_w x f_ch
      Thus, every batch has the same value
      -> output[0,x,y,z]==output[1,x,y,z]==output[n,x,y,z] for any x,y,and z    
    '''
    def __init__(self,tex_h,tex_w,f_ch,d_factor=0.5,**kwargs):
        self.tex_h=tex_h
        self.tex_w=tex_w
        self.f_ch=f_ch
        self.decay_factor=d_factor #should proportional to the tex_h,tex_w
        self.decay = np.sqrt(self.decay_factor**2+self.decay_factor**2)
        self.active_field=int(-np.log(0.1)*self.decay+ 0.5) #round up

        super(proj_to_tex_layer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(proj_to_tex_layer,self).build(input_shape)
    def call(self,x):
        f_map = x[0] #w,h,f_ch
        uv_map = x[1] #w,h,2
        mask = x[2]   #w,h,1
        pack_inputs =tf.concat([mask,uv_map,f_map],axis=3) #Batch x W x H, chs
        textures = tf.map_fn(self.project_to_texture,pack_inputs)        
        weighted_sum =tf.reduce_sum(textures[:,:,:,1:],axis=0,keep_dims=True) 
        weight_sum  = tf.reduce_sum(textures[:,:,:,:1],axis=0,keep_dims=True)
        weighted_sum = tf.cast(tf.greater(weight_sum,0.01),tf.float32)*weighted_sum
        unified_tex = weighted_sum / (weight_sum+0.001)        
        return tf.tile(unified_tex,[tf.shape(f_map)[0],1,1,1])
    
    def project_to_texture(self,packed_input):
        '''
        W,H -> size of an input image or a feature map
        input
        - uvs_map: rendered UV coordinates (Tensor: HxWx2 -> v/u order, tf.float32)
        - pixels: rendered(or real) RGB values (Tensor: HxWx3 -> r,g,b order, tf.float32)
        - masks: object_mask (Tensor: WxHx1 -> 1:object, 0:background, boolean)    
        - (opt) tex_h: height of the texture map (of the neural texture)
        - (opt) tex_w: width of the texture map (of the neural texture)
        - (opt) f_ch: number of channels of the texturemap (e.g.,3 for rgb textures)
        
        output
        - w_map: sum of accumulated weights of each pixels (Tensor: tex_h x tex_w x 1)
        - w_color: sum of accumulated weighted colors(or features) of each pixels (Tensor: tex_h x tex_w x ch, ch=3 for rgb images)
        e.g., final weighted average can be calculated by: norm_tex = w_color/(w_map+0.0001)
        '''
        #uvs_map,pixels,masks
        masks = packed_input[:,:,0]
        uvs_map = packed_input[:,:,1:3]
        pixels = packed_input[:,:,3:]
        tex_size= tf.constant([self.tex_h,self.tex_w],tf.float32)
        mask_indices = tf.where(tf.equal(masks,1))
        uvs_map = uvs_map[..., ::-1] 
        uvs_map = tf.multiply(uvs_map,tex_size)
        uv_gathered = tf.gather_nd(uvs_map,mask_indices)
        pix_gathered = tf.gather_nd(pixels,mask_indices)        
        
        weighted_maps = tf.zeros([self.tex_h, self.tex_w, 1+self.f_ch],tf.float32) #tf.constant(np.zeros((self.tex_h, self.tex_w, 1+self.f_ch)),tf.float32) #0:weight, 1,2,3:weighted_sum (r,g,b)         
        
        v_shift = tf.expand_dims(tf.maximum(uv_gathered[:,0]-self.active_field,0),axis=1) #Nx1
        u_shift = tf.expand_dims(tf.maximum(uv_gathered[:,1]-self.active_field,0),axis=1) #Nx1
        v_max = tf.expand_dims(tf.minimum(uv_gathered[:,0]+self.active_field,self.tex_h-1),axis=0)
        u_max = tf.expand_dims(tf.minimum(uv_gathered[:,1]+self.active_field,self.tex_w-1),axis=0)     
        u_grid,v_grid = tf.meshgrid(tf.range(0,2*self.active_field),tf.range(0,2*self.active_field))
        v_grid = tf.tile(tf.expand_dims(tf.cast(v_grid,tf.float32),axis=0),[tf.shape(v_shift)[0],1,1]) + tf.expand_dims(v_shift,axis=2) #Nx4x4 + NX1
        u_grid = tf.tile(tf.expand_dims(tf.cast(u_grid,tf.float32),axis=0),[tf.shape(u_shift)[0],1,1]) + tf.expand_dims(u_shift,axis=2) #Nx4x4
        tex_u = tf.expand_dims(tf.expand_dims(uv_gathered[:,1],axis=1),axis=2) #Nx1
        tex_v = tf.expand_dims(tf.expand_dims(uv_gathered[:,0],axis=1),axis=2)#Nx1
        
        diff_u = tf.minimum(tf.abs(u_grid -tex_u ),tf.abs(tf.abs(u_grid - tex_u)-self.tex_w))
        diff_v = tf.minimum(tf.abs(v_grid -tex_v),tf.abs(tf.abs(v_grid - tex_v)-self.tex_h))                        
        dist_map = tf.sqrt(tf.square(diff_u)+tf.square(diff_v))
        weight_map = tf.exp(-dist_map/self.decay) #NX4x4                            
        weight_fin =tf.cast(tf.greater(weight_map,0.1),tf.float32)*weight_map#Nx4x4
        weighted_sum =  tf.expand_dims(weight_fin,axis=3)* tf.expand_dims(tf.expand_dims(pix_gathered,axis=1),axis=1)  #Nx4x4x1 * Nx1x1x3
        #v_grid[Nx4x4],u_grid[Nx4x4],weight_fin[Nx4x4], weighted_sum[Nx4x4x3]
        indices = tf.cast(tf.stack([tf.reshape(v_grid,[-1]),tf.reshape(u_grid,[-1])],axis=1),tf.int64)        
        #Nx2
        values_w = tf.reshape(weight_fin,[-1,1])
        values_w_sum = tf.reshape(weighted_sum,[-1,self.f_ch])
        values_concat = tf.concat([values_w,values_w_sum],axis=1) #Nx(1+f_ch)
        output = tf.tensor_scatter_nd_add(weighted_maps,indices,values_concat) 
        return output
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.tex_h,self.tex_w,self.f_ch]))

class proj_to_tex_layer_rnn(proj_to_tex_layer):
    '''
    Project feature map into the texture feature space using proposed weighted average.
    arg:tex_h,tex_w,f_ch #target texture res and channel, f_ch==channel of input feature map
    [input]
    uv_map = Batch x W x H x 2
    feature_map = Batch x W x H x f_ch
    mask = Batch x W x H x 1    
    [output]
    - a projected_texture_map after Weighted average over batch.
      Batch x tex_h x tex_w x f_ch
      Thus, every batch has the same value
      -> output[0,x,y,z]==output[1,x,y,z]==output[n,x,y,z] for any x,y,and z    
    '''
    def __init__(self,**kwargs):
        #self.tex_h=tex_h
        #self.tex_w=tex_w
        #self.f_ch=f_ch
        #self.decay_factor=d_factor #should proportional to the tex_h,tex_w
        #self.decay = np.sqrt(self.decay_factor**2+self.decay_factor**2)
        #self.active_field=int(-np.log(0.1)*self.decay+ 0.5) #round up
        super(proj_to_tex_layer_rnn,self).__init__(**kwargs)
    def build(self,input_shape):
        super(proj_to_tex_layer_rnn,self).build(input_shape)
    def call(self,x):
        f_map = x[0] #w,h,f_ch
        uv_map = x[1] #w,h,2
        mask = x[2]   #w,h,1
        pack_inputs =tf.concat([mask,uv_map,f_map],axis=3) #Batch x W x H, chs
        textures = tf.map_fn(self.project_to_texture,pack_inputs)  
        batch_avg =tf.reduce_sum(textures,axis=0,keep_dims=True)        
        output = tf.concat([batch_avg,textures],axis=0)
        #output -> 1 x(1:avg +batch) x w x h x 4 (weight/weighted_sum)
        return output #converted to the LSTM input
        
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.tex_h,self.tex_w,self.f_ch+1]))


class neural_rendering_raster_and_texture(Layer):
    def __init__(self,img_h,img_w,focal,near=0.1,far=10.0,**kwargs):
        self.img_h=img_h
        self.img_w=img_w
        self.focal = focal
        self.near = near
        self.far = far        
        self.right = ((self.img_w-1.)/2.)*near/self.focal
        self.projection_matrix = matrices.perspective_projection(near=self.near, far=self.far, right=self.right, aspect=float(self.img_h) / self.img_w)    
        super(neural_rendering_raster_and_texture,self).__init__(**kwargs)
    def build(self,input_shape):
        super(neural_rendering_raster_and_texture,self).build(input_shape)
    def call(self,x):
        #N: num. of vertices, F: num of faces
        vertices = x[0] #(batchxNx3)
        uv_map = x[1] #(batchxNx2)
        faces = tf.cast(x[2],tf.int32) #Fx3
        texture = x[3]
        poses=x[4] #batchx4x4
        #inputs = (vertices_xyz,vertices_uv,faces,textures,poses)
        #vertices_xyz,vertices_uv,faces,textures,poses
        '''
        vertices: vertices (N,3) , tf.float32
        uv_map: uv_map (N,2), (normal GL convention) , tf.float32
        faces : faces (N,3) , tf.int32
        texture = (WxHx3 or C)  (0~1) texture image, tf.float32
        pose : (4x4)
        '''        
        # Transform vertices from camera to clip space        
        vertices_objects, vertices_cameras,vertices_clips=\
           tf.map_fn(self.transform_vertices,(vertices,poses),dtype=(tf.float32,tf.float32,tf.float32))        

        mask_uv = dirt.rasterise_batch(
                        background=tf.zeros([tf.shape(x[3])[0],self.img_h, self.img_w, 3]),
                        vertices=vertices_clips,
                        vertex_colors=tf.concat([
                            tf.ones_like(vertices_objects[:,:, :1]),  # mask
                            uv_map  # texture coordinates #cub e_normals_world,  # normals
                                    #cube_vertices_cameras[:,:, :3]
                        ], axis=2),
                        faces=faces,                                                
                        height=self.img_h,
                        width=self.img_w,
                        channels=3
                       )                
        
        mask_render = tf.expand_dims(mask_uv[:,:, :, 0],axis=3)
        unlit_colors = tf.map_fn(self.sample_texture,(texture, mask_uv[:,:,:,1:3] ),dtype=tf.float32)        
        #batc x h x w x 3   x batch x h x w x  
        pixels = unlit_colors * mask_render
        
        return pixels #tf.concat([pixels],axis=3)

    def transform_vertices(self,inputs):
        vertices = inputs[0]
        pose = inputs[1]
        cube_vertices_object = tf.concat([
            vertices,
            tf.ones_like(vertices[:, -1:])
        ], axis=1)
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
        return cube_vertices_object,cube_vertices_camera,cube_vertices_clip
    def uvs_to_pixel_indices(self,uvs, texture_shape):
        # Note that this assumes u = 0, v = 0 is at the top-left of the image -- different to OpenGL!
        uvs = uvs[..., ::-1]  # change x, y coordinates to y, x indices
        #batchxhxwx2  vs [2]
        texture_shape = tf.cast(texture_shape, tf.float32) #[h,w]
        return tf.clip_by_value(uvs, 0., 1.) * texture_shape        

    def sample_texture(self,inputs):
        texture=inputs[0]
        uv_val_ori=inputs[1] #0~1
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

    def shader_fn(self,gbuffer, texture):
            mask = gbuffer[:,:, :, :1]
            uvs = gbuffer[:,:, :, 1:3]
            #positions = gbuffer[:,:, :, 3:]
            unlit_colors = tf.map_fn(self.sample_texture,(texture, self.uvs_to_pixel_indices(uvs, tf.shape(texture)[:2]) ),dtype=tf.float32)
            pixels = unlit_colors * mask # + [0.] * (1. - mask)
            return pixels,mask#,positions            
    
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],self.img_h,self.img_w,4]))

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

class neural_rendering_raster_lighting(Layer):
    def __init__(self,img_h,img_w,cam_K,near=0.1,far=10.0,ch_dim=3,**kwargs):
        self.img_h=img_h
        self.img_w=img_w
        self.cam_K = cam_K
        self.near = near
        self.far = far        
        #self.right = ((self.img_w-1.)/2.)*near/self.focal
        self.projection_matrix = tf.constant(build_projection(cam_K,w=self.img_w,h=self.img_h),tf.float32)        
        self.ch_dim=ch_dim
        super(neural_rendering_raster_lighting,self).__init__(**kwargs)
    def build(self,input_shape):
        super(neural_rendering_raster_lighting,self).build(input_shape)
    def call(self,x):
        #N: num. of vertices, F: num of faces
        vertices = x[0] #(batchxNx3) #same for each batch
        uv_map = x[1] #(batchxNx2) #different for each batch
        faces = tf.cast(x[2],tf.int32) #batchxFx3, same for each batch
        texture = x[3] #batchxWxHxCH different for each batch
        poses=x[4] #batch x n_target_poses x 4x4, same for each batch
        #ignore batch dimension of poses
        vertices_mult = tf.tile(vertices,[tf.shape(poses)[1],1,1])
        vert_uvs_mult = tf.tile(uv_map,[tf.shape(poses)[1],1,1])
        faces_multi = tf.tile(faces,[tf.shape(poses)[1],1,1])
        texture_multi = tf.tile(texture,[tf.shape(poses)[1],1,1,1])        
        poses_t=tf.transpose(poses,[1,0,2,3])
        poses_t=tf.reshape(poses_t,[-1,4,4])

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
        rendered_feature = tf.map_fn(self.sample_texture,(texture_multi, gbuffer_temp[:,:,:,1:3] ),dtype=tf.float32)        
        #if both uv value is zero -> 
        #mask_old = gbuffer_temp[:,:,:,:1] #regardless of the facts that each pixel was seen by the input images
        mask_new = tf.cast(tf.greater(tf.reduce_sum(gbuffer_temp[:,:,:,1:3],axis=3,keepdims=True),0),tf.float32)
        rendered_feature = mask_new*rendered_feature #remove backgrounds
        concated_out = tf.concat([mask_new,rendered_feature],axis=3)
        final_out = tf.reshape(concated_out, [tf.shape(poses)[1],-1,self.img_h,self.img_w,self.ch_dim+1])
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
        return (tuple([input_shape[4][1],input_shape[0][0],self.img_h,self.img_w, self.ch_dim+1]))


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
