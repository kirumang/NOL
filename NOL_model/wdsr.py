'''
This code is modified from https://github.com/krasserm/super-resolution

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import tensorflow as tf

from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation,Layer
from keras.models import Model

from .common import SubpixelConv2D, Normalization, Denormalization


def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None,input_ch=3):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b,input_ch)

class resize_img(Layer):     
    def __init__(self,method='bilinear',**kwargs):            
        if(method=='bilinear'):
            self.method = tf.compat.v1.image.ResizeMethod.BILINEAR
            print("Decoder resizes using bi-linear")
        else:
            self.method = tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR
            print("Decoder resizes using nearest neighbor")
        super(resize_img,self).__init__(**kwargs)
    def build(self,input_shape):
        super(resize_img,self).build(input_shape)
    def call(self,x):
        img = x[0]        
        ref_size = x[1]    
        #4d-> corrrect input for the resize                            
        return tf.compat.v1.image.resize(img,[tf.shape(ref_size)[1],tf.shape(ref_size)[2]],method=self.method)
    def compute_output_shape(self,input_shape):        
        return (tuple([input_shape[0][0],input_shape[1][1],input_shape[1][2],input_shape[0][3]]))

def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block, input_ch):
    x_in = Input(shape=(None, None, input_ch))
    
    #x = Normalization()(x_in)

    # pad input if in test phase
    x = PadSymmetricInTestPhase()(x_in)
    
    
    m = Conv2D(num_filters, 3, padding='valid')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = Conv2D(3 * scale ** 2, 3, padding='valid', name='conv2d_main_scale_{}'.format(scale))(m)
    m = SubpixelConv2D(scale)(m)

    # skip branch
    s = Conv2D(3 * scale ** 2, 5, padding='valid', name='conv2d_skip_scale_{}'.format(scale))(x)
    s = SubpixelConv2D(scale)(s)
    

    x = Add()([m, s])
    x = resize_img(method='bilinear')([x,x_in])
    return Model(x_in, x, name="wdsr-b")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = Conv2D(num_filters * expansion, kernel_size, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = Conv2D(num_filters * expansion, 1, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(int(num_filters * linear), 1, padding='same')(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def PadSymmetricInTestPhase():
    pad = Lambda(lambda x: K.in_train_phase(x, tf.pad(x, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), 'SYMMETRIC')))
    pad.uses_learning_phase = True
    return pad
