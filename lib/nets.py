import tensorflow as tf
from . import arch

import numpy as np
import scipy.io
import utils

def vgg(path, x, center_data=True, pool_function='MAX'):
    mat = scipy.io.loadmat(path)
    mean_pixel = mat['meta'][0][0][2][0][0][2][0][0]
    net = x
    if center_data:
        net = x - mean_pixel

    pool_funcs = {'AVG': _avg_pool, 'MAX': _max_pool}

    net_layers = mat['layers'][0]
    layers= {}
    pool_num = 1
    relu_num = 1
    for layer_data in net_layers:
        layer = layer_data[0][0]
        layer_type = layer[1][0]
        if layer_type == 'conv':
            W_conv_data, b_conv_data = layer[2][0]
            # convert to TensorFlow representation
            # (height, width, in_chan, out_chan)
            W_conv_data = np.transpose(W_conv_data, (1, 0, 2, 3))
            # (n)
            b_conv_data = b_conv_data.reshape(-1)
            W_conv = tf.constant(W_conv_data)
            b_conv = tf.constant(b_conv_data)
            net = _conv2d(net, W_conv, 1) + b_conv
        elif layer_type == 'relu':
            net = tf.nn.relu(net)
            layers['relu'+str(pool_num)+'_'+str(relu_num)] = net
            relu_num += 1
        elif layer_type == 'pool':
            net = pool_funcs[pool_function](net, 2)
            pool_num += 1
            relu_num = 1

    return net, layers
