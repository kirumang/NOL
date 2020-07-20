import sys,os
import numpy as np
import scipy.ndimage
import tensorflow as tf
import tensorflow.keras


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten,Dense,Dropout,Activation,RepeatVector,Lambda,Reshape,Subtract
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Layer,TimeDistributed,Bidirectional
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2,Regularizer
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Add


import tensorflow_graphics as tfg
from tensorflow_graphics.geometry import transformation as tf_tf

import math
from tensorflow.keras.applications.densenet import DenseNet121
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from NOL_model.diff_renderer import neural_rendering_crop_resize,\
                                neural_rendering_gbuffer
from matplotlib import pyplot as plt
import NOL_model.tf_addon as tfa
from NOL_model import wdsr
class resize_uv_mask(Layer):     
    def __init__(self,mask=False,**kwargs):
        self.mask = mask
        super(resize_uv_mask,self).__init__(**kwargs)
    def build(self,input_shape):
        super(resize_uv_mask,self).build(input_shape)
    def call(self,x):
        img = x[0]
        ref_size = x[1]        
        if(self.mask):
            return tf.cast(tf.greater(tf.compat.v1.image.resize_bilinear(img,[tf.shape(ref_size)[1],tf.shape(ref_size)[2]]),0.5),tf.float32)
        else:
            return tf.compat.v1.image.resize_bilinear(img,[tf.shape(ref_size)[1],tf.shape(ref_size)[2]])            
    def compute_output_shape(self,input_shape):        
        return (tuple([input_shape[0][0],input_shape[1][1],input_shape[1][2],input_shape[0][3]]))

class resize_layer(Layer):     
    def __init__(self,width,height,**kwargs):
        self.width = width
        self.height = height
        super(resize_layer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(resize_layer,self).build(input_shape)
    def call(self,x):
        return tf.compat.v1.image.resize_bilinear(x,[self.height ,self.width])            
    def compute_output_shape(self,input_shape):        
        return (tuple([input_shape[0],self.height,self.width,input_shape[3]]))

class texture_loss(Layer):
    def __init__(self,weight=1.,**kwargs):
        self.weight=weight
        
        super(texture_loss,self).__init__(**kwargs)
    def build(self,input_shape):
        super(texture_loss,self).build(input_shape)
    def call(self,x):
        y_pred=x[0]
        y_gt=x[1] 
        texture_fm = x[2]
        visible = tf.reduce_sum(texture_fm[:,:,:,:3],axis=3,keepdims=True)
        visible = tf.cast(tf.greater(visible,0),y_pred.dtype)
        diff = tf.clip_by_value(tf.abs(y_gt-y_pred),0,1)
        loss_visible = visible*diff        
        loss = tf.clip_by_value(tf.reduce_sum(loss_visible,axis=[1,2,3])/(tf.reduce_sum(visible*3,axis=[1,2,3])+0.001),0,1)
        return self.weight*loss
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],1]))

class smooth_label_loss2(Layer):
    def __init__(self,**kwargs):
        super(smooth_label_loss2,self).__init__(**kwargs)
    def build(self,input_shape):
        super(smooth_label_loss2,self).build(input_shape)
    def call(self,x):
        '''
        Concept:okay, whatever label selected, keep close as possible with surrounding pixels
        1. Only Feature values 4: (0,1,2:color, 3:angles)
        2. 1st order gradient of feature values  #[l_scored_mask,rendered_mask,selected_cands])
        '''        
        #x vs x neighbors 
        #cond1: labels for surrounding pixel are diffrent 
        #cond2: minimize their feature difference (Gradient)
        soft_mask = x[0][:,:,:,:,0]  #[Pose X Batch X H X W x (1)]
        rendered_mask = x[1][:,:,:,:,0] #[Pose X batch x H X W x (1)]
        feature_raw = x[2] #PosexHxWx(C)        
        mask_as_feature = tf.transpose(soft_mask,[0,2,3,1]) #Pose X H X W x Batch -> soft_mask
        mask_per_img = tf.transpose(rendered_mask,[0,2,3,1])#Pose X H X W x Batch -> mask
        mask_per_pose = tf.cast(tf.greater(tf.reduce_sum(mask_per_img,axis=[3],keepdims=True),0),tf.float32)     
        #PoseXHxWx1
        
        #gradient
        grad_v = mask_per_pose[:,:-1,:,:]*feature_raw[:,:-1,:,:]-mask_per_pose[:,1:,:,:]*feature_raw[:,1:,:,:] #Pose x H-1 x W x C
        grad_u = mask_per_pose[:,:,:-1,:]*feature_raw[:,:,:-1,:]-mask_per_pose[:,:,1:,:]*feature_raw[:,:,1:,:] #Pose x H x W-1 x C        
        #Difference of gradient
        
        diff_v_grad_v = tf.abs(mask_per_pose[:,:-2,:,:]*grad_v[:,:-1,:,:]-mask_per_pose[:,:-2,:,:]*grad_v[:,1:,:,:])#Pose x H-2 x W x C
        diff_u_grad_v =  tf.abs(mask_per_pose[:,:-1,:-1,:]*grad_v[:,:,:-1,:]-mask_per_pose[:,:-1,:-1,:]*grad_v[:,:,1:,:])#Pose x H-1 x W-1 x C        
        diff_v_grad_u =  tf.abs(mask_per_pose[:,:-1,:-1,:]*grad_u[:,:-1,:,:]-mask_per_pose[:,:-1,:-1,:]*grad_u[:,1:,:,:]) #Pose x H-1 x W-1 x C
        diff_u_grad_u =  tf.abs(mask_per_pose[:,:,:-2,:]*grad_u[:,:,:-1,:]-mask_per_pose[:,:,:-2,:]*grad_u[:,:,1:,:]) #Pose x H x W-2 x C 
        
        loss1 = tf.reduce_sum(diff_v_grad_v,axis=[1,2,3])/(tf.reduce_sum(mask_per_pose[:,:-2,:,:])+1E-6)/tf.cast(tf.shape(feature_raw)[3],tf.float32)
        loss2 = tf.reduce_sum(diff_u_grad_v,axis=[1,2,3])/(tf.reduce_sum(mask_per_pose[:,:-1,:-1,:])+1E-6)/tf.cast(tf.shape(feature_raw)[3],tf.float32)
        loss3 = tf.reduce_sum(diff_v_grad_u,axis=[1,2,3])/(tf.reduce_sum(mask_per_pose[:,:-1,:-1,:])+1E-6)/tf.cast(tf.shape(feature_raw)[3],tf.float32)
        loss4 = tf.reduce_sum(diff_u_grad_u,axis=[1,2,3])/(tf.reduce_sum(mask_per_pose[:,:,:-2,:])+1E-6)/tf.cast(tf.shape(feature_raw)[3],tf.float32)
        loss = tf.reduce_mean((loss1+loss2+loss3+loss4)/4,keepdims=True)
        
        return tf.tile(loss,[tf.shape(x[1])[1]])
        
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[1][1]]))


class preprocess_img_layer(Layer):
    def __init__(self,**kwargs):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        super(preprocess_img_layer,self).__init__(**kwargs)
    def build(self,input_shape):
        super(preprocess_img_layer,self).build(input_shape)
    def call(self,x):        
        ch1=(x[:,:,:,:1]-self.mean[0])/self.std[0]
        ch2=(x[:,:,:,1:2]-self.mean[1])/self.std[1]
        ch3=(x[:,:,:,2:3]-self.mean[2])/self.std[2]
        out = tf.concat([ch1,ch2,ch3],axis=-1)
        return out
    def compute_output_shape(self,input_shape):
        return input_shape

class resize_uv_mask2(Layer):     
    def __init__(self,**kwargs):            
        super(resize_uv_mask2,self).__init__(**kwargs)
    def build(self,input_shape):
        super(resize_uv_mask2,self).build(input_shape)
    def call(self,x):
        img5d = x[0]
        img4d = img5d[:,0,:,:,:] #all pose x first batch x H,W,0
        ref_size = x[1]    
        #4d-> corrrect input for the resize                            
        return tf.cast(tf.greater(tf.compat.v1.image.resize_bilinear(img4d,[tf.shape(ref_size)[1],tf.shape(ref_size)[2]]),0.5),tf.float32)
    def compute_output_shape(self,input_shape):        
        return (tuple([input_shape[0][0],input_shape[1][1],input_shape[1][2],input_shape[0][4]]))
    
class perceptual_loss(Layer):
    def __init__(self,**kwargs):        
        super(perceptual_loss,self).__init__(**kwargs)
    def build(self,input_shape):
        super(perceptual_loss,self).build(input_shape)
    def call(self,x):
        f1_r= x[2]
        f2_r= x[3]
        #f3_r= x[6]
        #f4_r= x[7]
        
        f1_i= x[0][:tf.shape(f1_r)[0],:,:,:]
        f2_i= x[1][:tf.shape(f1_r)[0],:,:,:]
        #f3_i= x[2][:tf.shape(f1_r)[0],:,:,:]
        #f4_i= x[3][:tf.shape(f1_r)[0],:,:,:]

        mask_1= x[4]
        mask_2= x[5]
        #mask_3= x[10]
        #mask_4= x[11]

        ch_1= tf.cast(tf.shape(f1_i)[3],tf.float32)
        ch_2= tf.cast(tf.shape(f1_i)[3],tf.float32)
        ch_3= tf.cast(tf.shape(f1_i)[3],tf.float32)
        ch_4= tf.cast(tf.shape(f1_i)[3],tf.float32)        
        layer1 = tf.reduce_sum(mask_1*tf.square(f1_i-f1_r),axis=[1,2,3])/ (tf.reduce_sum(mask_1,axis=[1,2,3])*ch_1+0.001)
        layer2 = tf.reduce_sum(mask_2*tf.square(f2_i-f2_r),axis=[1,2,3])/ (tf.reduce_sum(mask_2,axis=[1,2,3])*ch_2+0.001)
        #layer3 = tf.reduce_sum(mask_3*tf.square(f3_i-f3_r),axis=[1,2,3])/ (tf.reduce_sum(mask_3,axis=[1,2,3])*ch_3+0.001)
        #layer4 = tf.reduce_sum(mask_4*tf.square(f4_i-f4_r),axis=[1,2,3])/ (tf.reduce_sum(mask_4,axis=[1,2,3])*ch_4+0.001)        
        loss_per_pose = (layer1+layer2)/ 2 #+layer3+layer4)/4 #n_posex1
        loss_single = tf.reduce_mean(loss_per_pose,keepdims=True) #[1,1]        
        return tf.tile(loss_single,[tf.shape(x[0])[0]])
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0]]))

class recont_loss(Layer):
    def __init__(self,**kwargs):
        super(recont_loss,self).__init__(**kwargs)
    def build(self,input_shape):
        super(recont_loss,self).build(input_shape)
    def call(self,x):
        visible = x[0]
        y_pred=x[1]
        y_gt=x[2][:tf.shape(y_pred)[0],:,:,:]                 
        visible = tf.cast(tf.greater(visible,0),y_pred.dtype)
        diff = tf.abs(y_gt-y_pred)
        loss_visible = visible*diff                
        loss = tf.reduce_sum(loss_visible,axis=[1,2,3])/(tf.reduce_sum(visible*3,axis=[1,2,3])+0.001)
        loss_single = tf.reduce_mean(loss,keepdims=True) #[1,1]                
        return tf.tile(loss_single,[tf.shape(x[2])[0]])

    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[2][0]]))


def map_vert_to_uv(cam_K):
    '''
    Derive projected UV map per image
    '''
    vert_xyz = Input(shape=(None,3))
    pose_train = Input(shape=(4,4)) #trans + quat (x,y,z, qx,qy,qz,qw])
    #pose_train = pose_vector_to_mat()(pose_vector) #batchx4x4
    vert_visible = Input(shape=(None,1))
    bbox = Input(shape=(4,)) #necessary to shift UV map to the given image space
    #transfrom vert_xyz using the given pose
    vert_tf = Lambda(lambda x: tf.transpose(tf.matmul( x[1][:,:3,:3],tf.transpose(x[0],[0,2,1]))+ x[1][:,:3,3:],[0,2,1]))\
              ([vert_xyz,pose_train])    
    #batch x N x 1
    u_temp = Lambda(lambda x:cam_K[0,0]*x[0][:,:,0]/(x[0][:,:,2])+cam_K[0,2]-x[1][:,1:2] )  ([vert_tf,bbox])
    v_temp = Lambda(lambda x:cam_K[1,1]*x[0][:,:,1]/(x[0][:,:,2])+cam_K[1,2]-x[1][:,0:1] )  ([vert_tf,bbox]) #v_min,u_min,v_max,u_max
    #batchx N x 1
    #batchx 1
     # (0) Invalid argument: Incompatible shapes: [15,8300,1] vs. [15,8300]
    u_temp = Lambda(lambda x: x[2]*  tf.expand_dims(x[0]/(x[1][:,3:4]-x[1][:,1:2]),axis=2)       )   ([u_temp,bbox,vert_visible])
    v_temp = Lambda(lambda x: x[2]*  tf.expand_dims(x[0]/(x[1][:,2:3]-x[1][:,0:1]),axis=2)      )([v_temp,bbox,vert_visible])
    #shift to the bbox and normalize
    #batchxNx1 + batchxNx1
    vert_uv = Concatenate()([u_temp,v_temp])    #if bbox is not 256 -> it is resized
    return Model(inputs=[vert_xyz,pose_train,vert_visible,bbox],outputs=[vert_uv],name='uv_projection')


def Render_and_integrate(ch_dim,ch_int=16,render_im_h=480,render_im_w=640,cam_K=np.eye(3),
                          target_im_w=256,target_im_h=256 ):
    texture_feature = Input(shape=(None,None,ch_dim))    
    vert_xyz = Input(shape=(None,3))
    vert_uv = Input(shape=(None,2))
    faces = Input(shape=(None,3))
    poses= Input(shape=(None,4,4))
    bboxes = Input(shape=(None,4))
    rendered = neural_rendering_crop_resize(img_h=render_im_h, img_w=render_im_w ,
                                            target_w=target_im_w,target_h = target_im_h,
                                            cam_K=cam_K,name="rendering",ch_dim=ch_dim)\
                                ([vert_xyz,vert_uv,faces,texture_feature,poses,bboxes])
    
    #rendered: pose X render from each image x H x W x ch(mask+channels)
    rendered_mask,rendered_feat,rendered_mask_raw,uv_projection = Lambda(lambda x : tf.split(x,[1,ch_dim,1,2],axis=4))(rendered)
    #rendered_mask: pose X render from each image x H x W x 1(mask)
    #rendered_feat: pose X render from each image x H x W x ch(channels)
    mask_0= Lambda(lambda x :tf.cast( tf.greater(tf.reduce_sum(x,axis=1),0),tf.float32))(rendered_mask)#pose x H x W x 1       
    rendered_raw = Lambda(lambda x : tf.transpose(x,[1,0,2,3,4]))(rendered_feat) #batch,pose, h x w x ch
    rendered_raw = Lambda(lambda x : x[:,:,:,:,:3])(rendered_raw)
    return Model(inputs=[vert_xyz,vert_uv,faces,texture_feature,poses,bboxes],outputs=[rendered_mask,rendered_feat,rendered_raw,mask_0,
                         rendered_mask_raw,uv_projection])                         

def integrated_img(ch_dim):
    #Compute intergrate features (currently 5D tensor)
    rendered_feat = Input(shape=(None,None,None,ch_dim))    
    #accumulate renderings from different poses.
    lstm1 = ConvLSTM2D(32,kernel_size = (3, 3),padding='same',return_sequences = True)(rendered_feat)
    lstm1 = ConvLSTM2D(32,kernel_size = (3, 3),padding='same',return_sequences = True)(lstm1)
    lstm_result = ConvLSTM2D(16,kernel_size = (3, 3),padding='same')(lstm1) #pose x H x W x 16
    
    lstm_feature = Lambda(lambda x : tf.tile(tf.expand_dims(x[0],axis=1),
                                            [1,tf.shape(x[1])[1],1,1,1]))\
                                            ([lstm_result,rendered_feat])
    compare_featues = Concatenate()([rendered_feat,lstm_feature]) #[Pose X Batch X H X W x 31] 
    
    return Model(inputs=[rendered_feat],outputs=[compare_featues])

def score_module(ch_dim,ch_int):
        compare_featues = Input(shape=(None,None,None,ch_dim+ch_int))            
        rendered_feat = Input(shape=(None,None,None,None))            
        
        l_score = TimeDistributed( Conv2D(32, (3, 3), name='score_conv1',padding='same',activation='relu'))(compare_featues) 
        l_score = TimeDistributed( Conv2D(16, (3, 3), name='score_conv2',padding='same',activation='relu'))(l_score) 
        l_score = TimeDistributed( Conv2D(8, (3, 3), name='score_conv3',padding='same',activation='relu'))(l_score) 
        l_score = TimeDistributed( Conv2D(1, (3, 3), name='score_conv5',padding='same')) (l_score) 
        l_scored_mask = Lambda(lambda x:tf.keras.activations.softmax(x,axis=1))(l_score)   
        selected_cands = Lambda(lambda x:tf.reduce_sum(x[0]*x[1],axis=1))([rendered_feat,l_scored_mask]) #01.24
        return Model(inputs=[compare_featues,rendered_feat],outputs=[selected_cands,l_scored_mask])
   

class inter_projection_error(Layer):
        def __init__(self,**kwargs):
            super(inter_projection_error,self).__init__(**kwargs)
        def build(self,input_shape):
            super(inter_projection_error,self).build(input_shape)
        def call(self,x):
            r_mask = x[0] # proj(ref_imgs) x1 x H x W x 1 
            r_proj = x[1]  #proj(ref_imgs) x 1x  x H x W x ch
            r_proj = tf.concat([r_proj[:,:,:,:,:3],r_proj[:,:,:,:,4:]],axis=-1)            

            shared_mask = r_mask[0] * r_mask[1:]  #img-1 x1 x H x W x1 #ECCV2020
            #shared_mask = r_mask[1:]  #img-1 x1 x H x W x1 #vert optimization
            proj_diff = r_proj[0] - r_proj[1:] #img-1 x1 x H x W x ch

            diff_on_shared_mask = tf.reduce_sum(tf.abs(shared_mask*proj_diff),axis=[1,2,3,4]) /\
                                 (tf.reduce_sum(shared_mask,axis=[1,2,3,4])+0.00001) #img-1

            return tf.concat([tf.reduce_sum(diff_on_shared_mask,axis=0,keepdims=True),diff_on_shared_mask],axis=0)
        def compute_output_shape(self,input_shape):
            return (tuple([input_shape[0][0]]))

class simple_mask_error(Layer):
        def __init__(self,**kwargs):
            super(simple_mask_error,self).__init__(**kwargs)
        def build(self,input_shape):
            super(simple_mask_error,self).build(input_shape)
        def call(self,x):
            r_buffer = tf.cast(x[0][:,:,:,0],tf.float32) #batchxHxWx11 (0)
            gt_img = tf.cast(x[1][:,:,:,0],tf.float32)
            block_error =tf.cast(x[1][:,:,:,1],tf.float32) #occ_mask
            return tf.reduce_mean((1-block_error)*tf.abs(r_buffer-gt_img),axis=[1,2])
        def compute_output_shape(self,input_shape):
            return (tuple([input_shape[0][0]]))

class simple_mask_error(Layer):
        def __init__(self,**kwargs):
            super(simple_mask_error,self).__init__(**kwargs)
        def build(self,input_shape):
            super(simple_mask_error,self).build(input_shape)
        def call(self,x):
            r_buffer = tf.cast(x[0][:,:,:,0],tf.float32) #batchxHxWx11 (0)
            gt_img = tf.cast(x[1][:,:,:,0],tf.float32)
            block_error =tf.cast(x[1][:,:,:,1],tf.float32) #occ_mask
            return tf.reduce_mean((1-block_error)*tf.abs(r_buffer-gt_img),axis=[1,2])
        def compute_output_shape(self,input_shape):
            return (tuple([input_shape[0][0]]))        
class iou_error(Layer):
        def __init__(self,**kwargs):
            super(iou_error,self).__init__(**kwargs)
        def build(self,input_shape):
            super(iou_error,self).build(input_shape)
        def call(self,x):
            r_buffer = tf.cast(x[0][:,:,:,0],tf.float32) #batchxHxWx11 (0)
            gt_img = tf.cast(x[1][:,:,:,0],tf.float32)
            block_error =tf.cast(x[1][:,:,:,1],tf.float32) #occ_mask
            
            intersect = (1-block_error)*tf.multiply(r_buffer,gt_img)
            union = (1-block_error)*tf.minimum(r_buffer+gt_img,1)
            return 1-tf.reduce_sum(intersect,axis=[1,2])/(tf.reduce_sum(union,axis=[1,2])+1E-6)
        def compute_output_shape(self,input_shape):
            return (tuple([input_shape[0][0]]))                 


class so3_to_mat(Layer):
    def __init__(self,**kwargs):
        gen1 = tf.constant(np.array([[[0,0,0],[0,0,-1],[0,1,0]]]),tf.float32)
        gen2 = tf.constant(np.array([[[0,0,1],[0,0,0],[-1,0,0]]]),tf.float32)
        gen3 = tf.constant(np.array([[[0,-1,0],[1,0,0],[0,0,0]]]),tf.float32)
        self.gen = tf.expand_dims(tf.concat([gen1,gen2,gen3],axis=0),axis=0) #1X3x3x3
        self.last_row = tf.constant(np.array([[[0,0,0,1]]]),tf.float32)
        super(so3_to_mat,self).__init__(**kwargs)
    def build(self,input_shape):
        super(so3_to_mat,self).build(input_shape)
    def call(self,x):
        w_rot = x[0][:,:3] # w1,w2,w3:rotation / w4,w5,w6 : translation
        w_tra = x[0][:,3:6] #Nx1x3
        w_rot = tf.expand_dims(tf.expand_dims(w_rot,axis=2),axis=3) #Nx3 -> Nx3x1x1        
        so3 = tf.reduce_sum(self.gen*w_rot,axis=1)        
        mat_rot = tf.linalg.expm(so3) #Nx3x3
        w_tra = tf.expand_dims(w_tra,axis=2) #Nx3 -> Nx3x1 ; Nx1x4
        mat3x4 = tf.concat([mat_rot,w_tra],axis=2) #Nx3x3 (c) Nx3x1 = Nx3x4
        last_rows = tf.tile(self.last_row,[tf.shape(w_rot)[0],1,1]) #1x1x4 -> Nx1x4        
        return tf.concat([mat3x4,last_rows],axis=1)
    
    def compute_output_shape(self,input_shape):
        return (tuple([input_shape[0][0],4,4]))


def neural_obj_learning(cam_K,render_im_w = 640,render_im_h = 480,
                       target_im_w=-1,target_im_h=-1):
    input_img = Input(shape=(None,None,3))    
    input_ang = Input(shape=(None,None,1))    
    
    vert_xyz = Input(shape=(None,3))
    faces = Input(shape=(None,3))
    
    vert_visible = Input(shape=(None,1))
    bbox_train= Input(shape=(4,)) 
    pose_input = Input(shape=(4,4))
    
    poses= Input(shape=(None,4,4))
    bboxes = Input(shape=(None,4))
    gt_render = Input(shape=(None,None,3))    
    gt_real_fake=Input(shape=(1,))
    
    feed_uv = map_vert_to_uv(cam_K)

    densenet=DenseNet121(include_top=False,weights='imagenet',input_shape=(None,None,3))
    backbone = Model(inputs=densenet.input, 
                     outputs=[densenet.get_layer('conv1/relu').output,
                              densenet.get_layer('pool2_conv').output,
                              densenet.get_layer('pool3_conv').output,
                              densenet.get_layer('pool4_conv').output])

    f1,f2,f3,f4=backbone(input_img) #size of f4 = input_dim/16
    f1_e = BatchNormalization(axis=3)(f1)
    f1_e = Activation('relu')(f1_e)
    f1_e = Conv2D(4, (3, 3), name='conv1_reduce2',padding='same')(f1_e)
    f1_e = BatchNormalization(axis=3)(f1_e)
    f1_e = Activation('relu')(f1_e) 
    
    f2_e = BatchNormalization(axis=3)(f2)
    f2_e = Activation('relu')(f2_e)
    f2_e = Conv2D(3, (3, 3), name='conv2_reduce2',padding='same')(f2_e)
    f2_e = BatchNormalization(axis=3)(f2_e)
    f2_e = Activation('relu')(f2_e)

    f3_e = BatchNormalization(axis=3)(f3)
    f3_e = Activation('relu')(f3_e)
    f3_e = Conv2D(3, (3, 3), name='conv3_reduce2',padding='same')(f3_e)
    f3_e = BatchNormalization(axis=3)(f3_e)
    f3_e = Activation('relu')(f3_e)

    f4_e = BatchNormalization(axis=3)(f4)
    f4_e = Activation('relu')(f4_e)
    f4_e = Conv2D(3, (3, 3), name='conv4_reduce2',padding='same')(f4_e)
    f4_e = BatchNormalization(axis=3)(f4_e)
    f4_e = Activation('relu')(f4_e)
    
    f1_ch =resize_uv_mask()([f1_e,input_img]) #4
    f2_ch =resize_uv_mask()([f2_e,input_img]) #3
    f3_ch =resize_uv_mask()([f3_e,input_img]) #3
    f4_ch =resize_uv_mask()([f4_e,input_img]) #3
    texture_feature = Concatenate()([input_img,input_ang,f1_ch,f2_ch,f3_ch,f4_ch]) 
    ch_dim = 3 + 1 + 4 + 3 + 3+ 3 #17
    
    vert_uv = feed_uv([vert_xyz,pose_input,vert_visible,bbox_train])
    render_featuremap = Render_and_integrate(ch_dim, ch_int=16,
                        render_im_h=render_im_h,render_im_w=render_im_w,cam_K=cam_K,
                        target_im_w=target_im_w,target_im_h=target_im_h)
    int_image_module = integrated_img(ch_dim)
    
    ##General path
    rendered_mask,rendered_feat,rendered_raw,mask_0,mask_raw,uv_projection=\
        render_featuremap([vert_xyz,vert_uv,faces,texture_feature,poses,bboxes])
    compare_featues = int_image_module(rendered_feat)    
    l_score = TimeDistributed( Conv2D(32, (3, 3), name='score_conv1',padding='same',activation='relu'))(compare_featues) 
    l_score = TimeDistributed( Conv2D(16, (3, 3), name='score_conv2',padding='same',activation='relu'))(l_score) 
    l_score = TimeDistributed( Conv2D(8, (3, 3), name='score_conv3',padding='same',activation='relu'))(l_score) 
    l_score = TimeDistributed( Conv2D(1, (3, 3), name='score_conv5',padding='same')) (l_score)  #PosexBatchxHxWx1
    l_scored_mask = Lambda(lambda x:tf.keras.activations.softmax(x,axis=1))(l_score)   
    selected_cands = Lambda(lambda x:tf.reduce_sum(x[0]*x[1],axis=1))([rendered_feat,l_scored_mask]) #01.24 #posexHxWx3
    
    #For tuning and intermediate outputs for alternative training process
    mask_output = Lambda(lambda x:tf.transpose(x,[1,0,2,3,4]))(rendered_mask) 
    proj_output = Lambda(lambda x:tf.transpose(x,[1,0,2,3,4]))(rendered_feat)     
    score_output = Lambda(lambda x:tf.transpose(x,[1,0,2,3,4]))(l_scored_mask)     
    mask_raw = Lambda(lambda x:tf.transpose(x,[1,0,2,3,4]))(mask_raw)     
    uv_projection= Lambda(lambda x:tf.transpose(x,[1,0,2,3,4]))(uv_projection)     
    int_error = inter_projection_error()([mask_output,proj_output])    
    

    decode_net = wdsr.wdsr_b(scale=2,
                            num_filters=32,
                            num_res_blocks=6, 
                            res_block_expansion = 4,
                            input_ch=ch_dim) 

    img_recont = decode_net(selected_cands) #pose x H x W x ch_dim (or 3?)
    img_recont_masked = Lambda(lambda x :x[0]*x[1])([mask_0,img_recont])
    img_recont_norm = preprocess_img_layer()(img_recont_masked)
    
    f1_i,f2_i,f3_i,f4_i=backbone(gt_render) #preprocessed, batch x H x W x 3 #batch>pose
    f1_r,f2_r,f3_r,f4_r=backbone(img_recont_norm) #-1~1 (also okay) 

    mask_1= resize_uv_mask(mask=True)([mask_0,f1_i]) #pose x H1 x W1 x 1 
    mask_2= resize_uv_mask(mask=True)([mask_0,f2_i]) #pose x H2 x W2 x 1 

    sml_loss = smooth_label_loss2()([l_scored_mask,rendered_mask,selected_cands])
    p_loss = perceptual_loss()([f1_i,f2_i,f1_r,f2_r,mask_1,mask_2])        
    r_loss = recont_loss()([mask_0,img_recont_norm,gt_render])

    Tuning_pose=Model(inputs=[input_img,input_ang,vert_xyz,faces,
                            pose_input,vert_visible,bbox_train,poses,bboxes],
                    outputs=[int_error,mask_output,proj_output,score_output,mask_raw])

    train_model = Model(inputs=[input_img,input_ang, #img info
                               vert_xyz,faces, #model info
                               pose_input,vert_visible,bbox_train, #input pose
                               poses,bboxes, #target img info
                               gt_render],
                        outputs=[r_loss,p_loss,sml_loss])
    
    l_score_output = Lambda(lambda x:tf.transpose(x,[1,0,2,3,4]),name='score_pred')(l_scored_mask) 
    img_recont_batch = Lambda(lambda x:tf.tile(x[0],[tf.shape(x[1])[0],1,1,1]),name='recont_pred')([img_recont,input_img])
    img_recont_norm_batch = Lambda(lambda x:tf.tile(x[0],[tf.shape(x[1])[0],1,1,1]))([img_recont_norm,input_img])
    render_gan = Model(inputs=[input_img,input_ang, #img info
                               vert_xyz,faces, #model info
                               pose_input,vert_visible,bbox_train, #input pose
                               poses,bboxes],
                        outputs=[img_recont_norm_batch,int_error,mask_output,
                                 proj_output,score_output,
                                 texture_feature,uv_projection])    
    #first after the rendering
    return train_model,backbone,render_gan,Tuning_pose 

def simple_render(img_h,img_w,cam_K):        
    vert_xyz = Input(shape=(None,3))
    faces = Input(shape=(None,3))    
    poses= Input(shape=(4,4))
    rendered = neural_rendering_gbuffer(img_h=img_h, img_w=img_w, cam_K=cam_K)\
               ([vert_xyz,faces,poses])
    return Model(inputs=[vert_xyz,faces,poses],outputs=[rendered])
