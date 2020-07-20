'''
This code is modified from https://github.com/krasserm/super-resolution

Copyright [2018] [Martin Krasser, https://github.com/krasserm/super-resolution]

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


import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Lambda

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def Normalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)


def Denormalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: x * 127.5 + rgb_mean, **kwargs)
