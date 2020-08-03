import os,sys
import numpy as np
import cv2
from skimage.filters import gaussian
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d
from scipy.spatial import cKDTree
from scipy import ndimage
import scipy.sparse
from skimage import morphology
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

import transforms3d as tf3d
import copy
import math
from tensorflow.keras.applications.densenet import preprocess_input
import math
from scipy.linalg import logm,expm

def pose_dist(x,y):
    '''
    rotational distance based on the inner product of quaternions
    '''
    dist = np.arccos(np.abs(np.sum(x*y))/ (np.linalg.norm(x)*np.linalg.norm(y)+0.00001) )
    return dist

def getXYZ(depth,fx,fy,cx,cy,bbox=np.array([0])):
     #get x,y,z coordinate in mm dimension
    uv_table = np.zeros((depth.shape[0],depth.shape[1],2),dtype=np.int16)
    column = np.arange(0,depth.shape[0])
    uv_table[:,:,1] = np.arange(0,depth.shape[1]) - cx #x-c_x (u)
    uv_table[:,:,0] = column[:,np.newaxis] - cy #y-c_y (v)

    if(bbox.shape[0]==1):
         xyz=np.zeros((depth.shape[0],depth.shape[1],3)) #x,y,z
         xyz[:,:,0] = uv_table[:,:,1]*depth*1/fx
         xyz[:,:,1] = uv_table[:,:,0]*depth*1/fy
         xyz[:,:,2] = depth
    else: #when boundry region is given
         xyz=np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1],3)) #x,y,z
         xyz[:,:,0] = uv_table[bbox[0]:bbox[2],bbox[1]:bbox[3],1]*depth[bbox[0]:bbox[2],bbox[1]:bbox[3]]*1/fx
         xyz[:,:,1] = uv_table[bbox[0]:bbox[2],bbox[1]:bbox[3],0]*depth[bbox[0]:bbox[2],bbox[1]:bbox[3]]*1/fy
         xyz[:,:,2] = depth[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    return xyz

def getRefinedDepth(depth,sigma=0.5):

    depth_refine = np.nan_to_num(depth)
    mask = np.zeros_like(depth_refine).astype(np.uint8)
    mask[depth_refine==0]=1
    depth_refine = depth_refine.astype(np.float32)
    depth_refine = cv2.inpaint(depth_refine,mask,2,cv2.INPAINT_NS)
    depth_refine = depth_refine.astype(np.float)
    depth_refine = gaussian(depth_refine,sigma=sigma)
    return depth_refine

def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def proj_2d_vu(xyz,fx,fy,cx,cy,res_x=640,res_y=480):
    u = ((fx*xyz[:,:,0]/xyz[:,:,2]+cx)).astype(np.int)
    v = ((fy*xyz[:,:,1]/xyz[:,:,2]+cy)).astype(np.int)
    return v,u

def proj_1d_vu(xyz,fx,fy,cx,cy,res_x=640,res_y=480):
    u = ((fx*xyz[:,0]/xyz[:,2]+cx)).astype(np.int)
    v = ((fy*xyz[:,1]/xyz[:,2]+cy)).astype(np.int)
    return v,u

def proj_2d(xyz,fx,fy,cx,cy,res_x=640,res_y=480):
    depth_proj = np.zeros((res_y,res_x))+1000
    u = ((fx*xyz[:,0]/xyz[:,2]+cx)).astype(np.int)
    v = ((fy*xyz[:,1]/xyz[:,2]+cy)).astype(np.int)

    for idx in range(xyz.shape[0]):
        z = xyz[idx,2]
        if(v[idx]<res_y and v[idx]>=0 and u[idx]>=0 and u[idx] < res_x):
            if(z < depth_proj[v[idx],u[idx]]):
                depth_proj[v[idx],u[idx]]=z

    depth_proj[depth_proj==1000]=0
    return depth_proj

def proj_2d_pt(pt,fx,fy,cx,cy,res_x=640,res_y=480):    
    u = fx*pt[0]/pt[2]+cx
    v = fy*pt[1]/pt[2]+cy
    return v,u

def proj_2d_mesh(xyz,faces,fx,fy,cx,cy,res_x=640,res_y=480):
    depth_proj = np.zeros((res_y,res_x))+1000
    u = ((fx*xyz[:,0]/xyz[:,2]+cx)).astype(np.int)
    v = ((fx*xyz[:,1]/xyz[:,2]+cy)).astype(np.int)

    for idx in range(faces.shape[0]):
        z = xyz[idx,2]
        if(v[idx]<res_y and v[idx]>=0 and u[idx]>=0 and u[idx] < res_x):
            if(z < depth_proj[v[idx],u[idx]]):
                depth_proj[v[idx],u[idx]]=z

    depth_proj[depth_proj==1000]=0
    return depth_proj

def get_normal(depth_refine,fx,fy,cx,cy):
    res_y=depth_refine.shape[0]
    res_x=depth_refine.shape[1]
    constant_x = 1/fx
    constant_y=  1/fy
    
    uv_table = np.zeros((res_y,res_x,2),dtype=np.int16)
    column = np.arange(0,res_y)
    uv_table[:,:,1] = np.arange(0,res_x) - cx #x-c_x (u)
    uv_table[:,:,0] = column[:,np.newaxis] - cy #y-c_y (v)

    v_x = np.zeros((res_y,res_x,3))
    v_y = np.zeros((res_y,res_x,3))
    normals = np.zeros((res_y,res_x,3))

    uv_table_sign= np.copy(uv_table)
    uv_table=np.abs(np.copy(uv_table))

    depth_refine = np.nan_to_num(depth_refine)
    mask = np.zeros_like(depth_refine).astype(np.uint8)
    mask[depth_refine==0]=1
    depth_refine = depth_refine.astype(np.float32)
    depth_refine = cv2.inpaint(depth_refine,mask,2,cv2.INPAINT_NS)
    depth_refine = depth_refine.astype(np.float)
    depth_refine = gaussian(depth_refine,sigma=2)
    
    dig=np.gradient(depth_refine,2,edge_order=2)
    v_y[:,:,0]=uv_table_sign[:,:,1]*constant_x*dig[0]
    v_y[:,:,1]=depth_refine*constant_y+(uv_table_sign[:,:,0]*constant_y)*dig[0]
    v_y[:,:,2]=dig[0]

    v_x[:,:,0]=depth_refine*constant_x+uv_table_sign[:,:,1]*constant_x*dig[1]
    v_x[:,:,1]=uv_table_sign[:,:,0]*constant_y*dig[1]
    v_x[:,:,2]=dig[1]

    cross = np.cross(v_x.reshape(-1,3),v_y.reshape(-1,3))
    norm = np.expand_dims(np.linalg.norm(cross,axis=1),axis=1)
    norm[norm==0]=1
    cross = cross/norm
    cross =cross.reshape(res_y,res_x,3)    
    cross= np.nan_to_num(cross)
    return cross,depth_refine
    

from skimage.transform import resize
def compute_pose_diff(pose_a,pose_b):
    q_a= tf3d.quaternions.mat2quat(pose_a[:3,:3])
    q_b = tf3d.quaternions.mat2quat(pose_b[:3,:3])
    min_diff = min(np.sum(q_a*q_b),np.sum(-q_a*q_b))
    return np.arccos(min_diff)

def compute_face_normals(vertices, faces):
    vertices_by_index = np.copy(vertices) # indexed by vertex-index, *, x/y/z
    vert_0 = vertices_by_index[faces[:,0]]
    vert_1 = vertices_by_index[faces[:,1]]
    vert_2 = vertices_by_index[faces[:,2]]
    normals_by_face = np.cross(vert_1 - vert_0, vert_2 - vert_0)  # indexed by face-index, *, x/y/z
    normals_by_face /= (np.linalg.norm(normals_by_face, axis=-1, keepdims=True) + 1.e-12)  # ditto            
    return normals_by_face

def get_sorted_cands(t_vert,t_faces,t_pose,t_poses,simple_renderer,t_vert_vis,n_img,incl_gt_uv=False,cam_K=-1,bbox=None,get_obj_3d=False,max_share=False):
    r_vert3d =np.zeros((1,t_vert.shape[0],3),np.float32)
    r_faces = np.zeros((1,t_faces.shape[0],3),np.int32)
    r_poses = np.zeros((1,4,4),np.float32)            
    r_vert3d[0]=t_vert
    r_faces[0]=t_faces
    r_poses[0]=t_pose
    rend_xyz = simple_renderer.predict([r_vert3d,r_faces,r_poses])
    rend_xyz =rend_xyz[0]
    camera_angle = rend_xyz[:,:,1:4] #HxWx3
    render_mask = rend_xyz[:,:,0]>0
    xyz = rend_xyz[render_mask,1:4]                    
    mesh_vert = (np.matmul(t_pose[:3,:3],np.copy(t_vert).T)+t_pose[:3,3:4]).T
    mesh_normal = compute_face_normals(mesh_vert, t_faces)
    vert_uv_img = np.zeros((t_vert.shape[0],2))
    depth_r = np.copy(rend_xyz[:,:,3]) #check in the previous version

    if incl_gt_uv:        
        bbox = bbox.astype(np.int)
        rend_normals,_ = get_normal(rend_xyz[:,:,3],cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2])        
        valid_xyz = np.linalg.norm(camera_angle,axis=2)
        valid_normal = np.linalg.norm(rend_normals,axis=2)
        mask_face = np.logical_and(np.logical_and(render_mask,valid_xyz>0),valid_normal>0)
        camera_angle = camera_angle[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        mask_face = mask_face[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        valid_normal = valid_normal[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        valid_xyz = valid_xyz[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        rend_normals = rend_normals[bbox[0]:bbox[2],bbox[1]:bbox[3]]        
        face_angle = np.zeros((bbox[2]-bbox[0],bbox[3]-bbox[1]))
        face_angle[mask_face] =np.sum(camera_angle[mask_face]*rend_normals[mask_face],axis=1)/valid_xyz[mask_face]/valid_normal[mask_face]
        face_angle = resize(face_angle,(256,256))        
    
    tf_vert = np.copy(mesh_vert.T)
    valid_indice = np.zeros((mesh_vert.shape[0]),bool)
    u_temp  =(cam_K[0,0]*tf_vert[0]/tf_vert[2] + cam_K[0,2]).astype(np.int)
    v_temp  =(cam_K[1,1]*tf_vert[1]/tf_vert[2] + cam_K[1,2]).astype(np.int)
    valid_idx = np.logical_and(u_temp>=0, u_temp+1<depth_r.shape[1])
    valid_idx = np.logical_and(np.logical_and(v_temp>=0, v_temp+1<depth_r.shape[0]),valid_idx)        
    valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx],u_temp[valid_idx]]-tf_vert[2,valid_idx])<0.01
    valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx]+1,u_temp[valid_idx]]-tf_vert[2,valid_idx])<0.01
    valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx],u_temp[valid_idx]+1]-tf_vert[2,valid_idx])<0.01
    valid_indice[valid_idx] = np.abs(depth_r[v_temp[valid_idx]+1,u_temp[valid_idx]+1]-tf_vert[2,valid_idx])<0.01
    #remove invalid faces
    visible_face_id = (valid_indice[t_faces[:,0]]+valid_indice[t_faces[:,1]]+valid_indice[t_faces[:,2]])>0                            
    invalid_normal_face_id = t_faces[np.logical_and(mesh_normal[:,2]>0,
                                                    np.invert(visible_face_id))]
    valid_indice[invalid_normal_face_id[:,0]]=0
    valid_indice[invalid_normal_face_id[:,1]]=0
    valid_indice[invalid_normal_face_id[:,2]]=0 
    
    if(incl_gt_uv):        
        vert_uv_img[valid_indice,0]  = (u_temp[valid_indice] -bbox[1])/(bbox[3]-bbox[1]-1)
        vert_uv_img[valid_indice,1]  = (v_temp[valid_indice] -bbox[0])/(bbox[2]-bbox[0]-1)
        vert_uv_img[np.logical_or(vert_uv_img<0,vert_uv_img>1)]=0
    
    #2) Loop over Vertex candidates <- shared portion
    indice_sum = np.zeros_like(valid_indice)
    #im_ids = np.arange(0,t_poses.shape[0]).tolist()
    im_ids = np.arange(0,t_poses.shape[0])
    sorted_ids=[]
    included_poses=[]
    #also prefers more diverse poses
    vert_vis_temp = np.copy(t_vert_vis)
    #Encourage to get a pose from different views
    #add a sample if target pose is the same
    n_iter=0

    while len(sorted_ids)<=n_img+1 and vert_vis_temp.shape[0]>0 : #in case of incl. self
        active_idx = np.ones((vert_vis_temp.shape[0]),bool)    
        after_visible = indice_sum+vert_vis_temp.astype(np.float32)
        vert_score = np.sum(np.tanh(1.5*after_visible[:,valid_indice]),axis=1)
        
        #t_pose,t_poses
        has_same_pose=False
        if(n_iter==0):
            for idx, im_id in enumerate(im_ids):
                if np.linalg.norm(t_pose[:3,3]-t_poses[im_id][:3,3])<0.001:
                    rx,ry,rz = tf3d.euler.mat2euler(np.matmul(np.linalg.inv(t_poses[im_id][:3,:3]), t_pose[:3,:3]))
                    if np.abs(rx)<0.001 and np.abs(ry)<0.001 and np.abs(rz)<0.001:
                        has_same_pose=True
                        best_idx = idx
        if n_iter>0 or not(has_same_pose):
            best_idx =np.argmax(vert_score)    
            indice_sum=indice_sum+(vert_vis_temp[best_idx]).astype(np.float32)                

        best_im_id = im_ids[best_idx]
        sorted_ids.append(best_im_id)        
        included_poses.append(t_poses[best_im_id])
        
        active_idx[best_idx]=0        
        im_ids = im_ids[active_idx]        
        vert_vis_temp = vert_vis_temp[active_idx]
        n_iter+=1
    if(get_obj_3d):
        return sorted_ids,render_mask,vert_uv_img,face_angle,rend_xyz[:,:,7:10]
    elif(incl_gt_uv):
        return sorted_ids,render_mask,vert_uv_img,face_angle
    else:
        return sorted_ids,render_mask

def get_tuning_set(t_id,t_images,t_poses,t_bboxes,t_vert,t_faces,simple_renderer,t_vert_vis,n_img,t_bboxes_ori,\
                   t_masks, t_face_angles,\
                   cam_K=None,res_x=-1,res_y=-1,bbox_ori=np.zeros((1)),
                   max_share=False
                   ):
        b_vert3d =np.zeros((n_img,t_vert.shape[0],3),np.float32)
        b_faces = np.zeros((n_img,t_faces.shape[0],3),np.int32)
        b_mask = np.zeros((n_img,256,256,1))
        b_poses = np.zeros((n_img,1,4,4),np.float32)            
        b_img =np.zeros((n_img,256,256,3),np.float32) #input image / output target
        b_img_gt =np.zeros((n_img,256,256,3),np.float32) #input image / output target      
        b_ang_in =np.zeros((n_img,256,256,1),np.float32) #input image / output target      

        b_vert3d[:]=t_vert
        b_faces[:]=t_faces
        b_bboxes = np.zeros((n_img,1,4),np.float32)            

        b_pose_input =np.zeros((n_img,4,4),np.float32) #input image / output target      
        b_vert_visible =np.zeros((n_img,t_vert.shape[0],1),np.float32) #input image / output target      
        b_bbox_train =np.zeros((n_img,4),np.float32) #input image / output target      
        if(len(t_id.shape)==0):            
            b_img_gt[:] = preprocess_input(np.copy(t_images[t_id])*255)         
            b_poses[:,0,:,:] = t_poses[t_id]
            b_bboxes[:,0,:]=t_bboxes[t_id]
            target_pose = tf3d.quaternions.mat2quat(t_poses[t_id][:3,:3])        
            sorted_ids,render_mask = get_sorted_cands(t_vert,t_faces,np.copy(t_poses[t_id]),t_poses,simple_renderer,t_vert_vis,n_img,
                                                            incl_gt_uv=False,max_share=max_share)
        else:
            target_pose = t_id            
            b_poses[:,0,:,:] = target_pose
            if(bbox_ori.shape[0]==1):
                bbox_ori = np.array([int(cam_K[1,2])-128,int(cam_K[0,2])-128,int(cam_K[1,2])+128,int(cam_K[0,2])+128])                
            bbox_temp =bbox_ori/np.array([res_y,res_x,res_y,res_x])
            b_bboxes[:,0,:]=bbox_temp
            sorted_ids,render_mask,uv_target,face_target,obj_3d = get_sorted_cands(t_vert,t_faces,target_pose,t_poses,simple_renderer,t_vert_vis,n_img,
                                                                            incl_gt_uv=True,cam_K=cam_K,bbox=bbox_ori,get_obj_3d=True,max_share=max_share) #to recover target uv values

        
        n_count=0
        im_ids=[]
        for b_id,im_id in enumerate(sorted_ids):
            img_processed = preprocess_input(np.copy(t_images[im_id]) *255 )    
            b_img[n_count] = img_processed                        
            b_mask[n_count] = t_masks[im_id]
            b_ang_in[n_count,:,:,0]=t_face_angles[im_id]             
            b_pose_input[n_count]=t_poses[im_id]
            b_bbox_train[n_count]=t_bboxes_ori[im_id]        
            b_vert_visible[n_count,:,0]=t_vert_vis[im_id].astype(np.float32)
            im_ids.append(im_id)
            n_count+=1    
            if(n_count>=n_img):break
        if(len(t_id.shape)==0):
            return b_img,b_ang_in,\
                   b_vert3d,b_faces,\
                   b_pose_input,b_vert_visible,b_bbox_train,\
                   b_poses,b_bboxes,b_img_gt
        else:
            return b_img,b_ang_in,\
                   b_vert3d,b_faces,\
                   b_pose_input,b_vert_visible,b_bbox_train,\
                   b_poses,b_bboxes,b_img_gt,uv_target,face_target,im_ids,obj_3d


def add_integrated_img(img_init,face_angle,target_pose,uv_target,bbox_ori,\
                       b_img,b_ang_in,b_vert3d,b_faces,b_pose_input,b_vert_visible,b_bbox_train,b_poses,b_bboxes):
    f_img = np.concatenate([np.expand_dims(img_init,axis=0),b_img],axis=0)
    f_ang_in= np.concatenate([np.expand_dims(np.expand_dims(face_angle,axis=0),axis=3),b_ang_in],axis=0)
    f_vert3d = np.concatenate([b_vert3d[:1],b_vert3d],axis=0)
    f_faces = np.concatenate([b_faces[:1],b_faces],axis=0)
    f_pose_input = np.concatenate([np.expand_dims(target_pose,axis=0),b_pose_input],axis=0)
    vert_vis_target = np.sum(uv_target,axis=1)>0
    f_vert_visible = np.concatenate([np.expand_dims(np.expand_dims(vert_vis_target,axis=0),axis=2).astype(np.float32),b_vert_visible],axis=0)
    #f_vert_visible = np.concatenate([np.expand_dims(vert_vis_target,axis=0).astype(np.float32),b_vert_visible],axis=0)
    f_bbox_train = np.concatenate([np.expand_dims(bbox_ori,axis=0),b_bbox_train],axis=0)
    f_poses = np.concatenate([b_poses[:1],b_poses],axis=0)
    f_bboxes = np.concatenate([b_bboxes[:1],b_bboxes],axis=0)
    return f_img,f_ang_in,f_vert3d,f_faces,f_pose_input,f_vert_visible,f_bbox_train,f_poses,f_bboxes


def get_edge_imgage(img_source,render_xyz,c_box,cam_K,th=0.1):
    normal_xyz,_ = get_normal(render_xyz[:,:,2],fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2])
    render_xyz = render_xyz[c_box[0]:c_box[2],c_box[1]:c_box[3]]
    normal_xyz = normal_xyz[c_box[0]:c_box[2],c_box[1]:c_box[3]]
    mask = render_xyz[:,:,2]>0
    dx_x,dx_y = np.gradient(normal_xyz[:,:,0])
    dy_x,dy_y = np.gradient(normal_xyz[:,:,1])
    dz_x,dz_y = np.gradient(normal_xyz[:,:,2])
    dist_x,dist_y = np.gradient(render_xyz[:,:,2])
    vx = np.array([0+dx_x*0,0+dy_x*0,1+dz_x*0])
    vy = np.array([0+dx_x,0+dy_x,1+dz_x])
    diff_x = np.arccos( np.sum(vx*vy,axis=0)/np.linalg.norm(vy,axis=0))
    vy = np.array([0+dx_y,0+dy_y,1+dz_y])
    diff_y = np.arccos( np.sum(vx*vy,axis=0)/np.linalg.norm(vy,axis=0))
    th=0.1
    depth_edge = np.logical_and(np.abs(dist_x)>0.1,np.abs(dist_y)>0.1)
    edge_mask = np.logical_and(np.logical_or(np.abs(diff_x)>th,np.abs(diff_y)>th),mask)
    edge_mask = np.logical_or(edge_mask,depth_edge)
    edge_mask = resize(edge_mask.astype(np.float32),(256,256))>0
    img_source[edge_mask>0]=[0,1,0]
    return img_source


def SO3_to_so3(mat):
    ln_R,_ = logm(mat,disp=False)
    w1= ln_R[2,1]
    w2= ln_R[0,2]
    w3= ln_R[1,0]
    return np.array([w1,w2,w3])

def so3_to_SO3(w):
    gen1 = np.array([[[0,0,0],[0,0,-1],[0,1,0]]])
    gen2 = np.array([[[0,0,1],[0,0,0],[-1,0,0]]])
    gen3 = np.array([[[0,-1,0],[1,0,0],[0,0,0]]])
    gen = np.concatenate([gen1,gen2,gen3],axis=0)
    w = np.expand_dims(w,axis=[1,2])
    so3 = np.sum(gen*w,axis=0)    
    return expm(so3)

def get_edge_imgage2(img_source,render_xyz,c_box,cam_K,th=0.1):
    render_xyz = render_xyz[c_box[0]:c_box[2],c_box[1]:c_box[3]]
    mask = render_xyz[:,:,2]>0
    dist_x,dist_y = np.gradient(render_xyz[:,:,2])
    depth_edge = np.logical_or(np.abs(dist_x)>th,np.abs(dist_y)>th)
    edge_mask = depth_edge
    edge_mask = resize(edge_mask.astype(np.float32),(256,256))>0
    edge_mask = binary_dilation(edge_mask)>0
    img_source[edge_mask>0]=[0,1,1]
    return img_source

def save_json(path, content):
    f= open(path, 'w')
    f.write('{\n')
    content_sorted = sorted(content.items(), key=lambda x: x[0])
    for elem_id, (k, v) in enumerate(content_sorted):
        f.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
        if elem_id != len(content) - 1:
            f.write(',')
        f.write('\n')
    f.write('}')   
    
def get_face_angles(depth_r,cam_K,im_height,im_width):
    mask_effect = depth_r>0
    xyz = getXYZ(depth_r,cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2])
    normal2,_ = get_normal(depth_r,cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2])
    normal2[np.invert(mask_effect)]=0
    
    camera_angle = np.copy(xyz)
    face_angle2= np.zeros( (im_height,im_width))
    valid_xyz = np.linalg.norm(camera_angle,axis=2)
    valid_normal = np.linalg.norm(normal2,axis=2)
    mask_face = np.logical_and(np.logical_and(depth_r>0,valid_xyz>0),valid_normal>0)
    
    face_angle2[mask_face] = np.sum(camera_angle[mask_face]*normal2[mask_face],axis=1)\
                                    /valid_xyz[mask_face]/valid_normal[mask_face]
    return face_angle2   

def get_valid_vertices(c_vert,face_obj,depth_r,pose,cam_K,im_width,im_height,mask=None):       
    '''
    Compute visible vertices in the pose (using a rendered detph image in the pose without occlusion)
    If mask is given, vertices that do not belong to the mask are regarded as invisible vetices. 
    '''
    mask_ren = depth_r>0
    vert_vis = np.ones((c_vert.shape[0]))    
    tf_vert = (np.matmul(pose[:3,:3],c_vert.T)+pose[:3,3:4])
    mesh_vert = np.copy(tf_vert.T)    
    mesh_normal = compute_face_normals(mesh_vert,face_obj)        
    xyz = getXYZ(depth_r,cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2])
    
    xyz=xyz[mask_ren]
    valid_indice = np.zeros((mesh_vert.shape[0]),bool)           
    
    nn = NearestNeighbors(n_neighbors=50).fit(mesh_vert)                    
    dists,indices = nn.kneighbors(xyz)
    vert_mask = np.zeros((mesh_vert.shape[0]),np.uint8)
    valid_indice = np.zeros((mesh_vert.shape[0]),bool)
    for pt_idx in range(xyz.shape[0]):
        valid_ver = indices[pt_idx,dists[pt_idx,:]<0.01]                
        vert_mask[valid_ver]=1            
    visible_face_id = (vert_mask[face_obj[:,0]]+vert_mask[face_obj[:,1]]+vert_mask[face_obj[:,2]])>0                    
    valid_normal_face_id = mesh_normal[:,2]<=0
    valid_faces =face_obj[np.logical_and(visible_face_id,valid_normal_face_id)]
    valid_indice[valid_faces[:,0]]=1
    valid_indice[valid_faces[:,1]]=1
    valid_indice[valid_faces[:,2]]=1 

    if(mask is not None):
        #use mask if objects are partially occluded by other objects
        u_temp  =(cam_K[0,0]*tf_vert[0]/tf_vert[2] + cam_K[0,2]).astype(np.int)
        v_temp  =(cam_K[1,1]*tf_vert[1]/tf_vert[2] + cam_K[1,2]).astype(np.int)
        valid_idx = np.logical_and(u_temp>=0, u_temp+1<im_width)
        valid_idx = np.logical_and(np.logical_and(v_temp>=0, v_temp+1<im_height),valid_idx)    

        valid_uvs = np.zeros_like(valid_indice)  
        valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],
                                             mask[v_temp[valid_idx],u_temp[valid_idx]])
        valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],
                                             mask[v_temp[valid_idx]+1,u_temp[valid_idx]])
        valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],
                                             mask[v_temp[valid_idx],u_temp[valid_idx]+1])
        valid_uvs[valid_idx] = np.logical_or(valid_uvs[valid_idx],
                                             mask[v_temp[valid_idx]+1,u_temp[valid_idx]+1])
        valid_indice[valid_idx] = np.logical_and(valid_indice[valid_idx],valid_uvs[valid_idx])  

    return valid_indice

