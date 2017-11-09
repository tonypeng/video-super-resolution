import numpy as np
import scipy.io
import tensorflow as tf
from . import arch, utils

_STD_DEV = 0.1

def MFSR(p,x):
    # input p is the prediction, x is the original low quality video
    FilmSize = utils.tensor_shape(x)
    Input = np.zeros(FilmSize)
    for i in range(FilmSize[0]):
        x[i] = arch.spatial_replication_padding(x[i], 1, utils.tensor_shape(x[i]), (9, 9))
        p[i] = arch.spatial_replication_padding(p[i], 1, utils.tensor_shape(p[i]), (9, 9))
    for i in range(FilmSize[0]):
        # concat 3 different frames into 9 channels
        if i == 0 :
            Input[i] = tf.concat([p[i],x[i],p[i+1]],2)
        else if i == FilmSize[0]-1:
            Input[i] = tf.concat([p[i-1],x[i],p[i]],2)
        else:
            Input[i] = tf.concat([p[i-1],x[i],p[i+1]],2)

    chan_conv1 = 32
    W_conv1 = arch.initialize_weights([9, 9, 9, chan_conv1], _STD_DEV)
    conv1 = arch.conv2d(Input, W_conv1, 1, border_mode='VALID')
    conv1 = arch.instance_normalization(conv1, chan_conv1)
    act1 = tf.nn.elu(conv1)

    chan_conv2 = 64
    W_conv2 = arch.initialize_weights([3, 3, chan_conv1, chan_conv2], _STD_DEV)
    conv2 = arch.conv2d(act1, W_conv2, 2)
    conv2 = arch.instance_normalization(conv2, chan_conv2)
    act2 = tf.nn.elu(conv2)

    chan_conv3 = 128
    W_conv3 = arch.initialize_weights([3, 3, chan_conv2, chan_conv3], _STD_DEV)
    conv3 = arch.conv2d(act2, W_conv3, 2)
    conv3 = arch.instance_normalization(conv3, chan_conv3)
    act3 = tf.nn.elu(conv3)

    W_resid1_1, W_resid1_2 = (
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV))
    resid1 = arch.resid_block(act3, W_resid1_1, W_resid1_2, tf.nn.elu, chan_conv3)
    W_resid2_1, W_resid2_2 = (
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV))
    resid2 = arch.resid_block(resid1, W_resid2_1, W_resid2_2, tf.nn.elu, chan_conv3)
    W_resid3_1, W_resid3_2 = (
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV))
    resid3 = arch.resid_block(resid2, W_resid3_1, W_resid3_2, tf.nn.elu, chan_conv3)
    W_resid4_1, W_resid4_2 = (
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV))
    resid4 = arch.resid_block(resid3, W_resid4_1, W_resid4_2, tf.nn.elu, chan_conv3)
    W_resid5_1, W_resid5_2 = (
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3], _STD_DEV))
    resid5 = arch.resid_block(resid4, W_resid5_1, W_resid5_2, tf.nn.elu, chan_conv3)

    chan_deconv1 = 64
    W_deconv1 = arch.initialize_weights([3, 3, chan_deconv1, chan_conv3], _STD_DEV)
    deconv1 = arch.resizeconv2d(resid5, W_deconv1, 2)
    deconv1 = arch.instance_normalization(deconv1, chan_deconv1)
    act4 = tf.nn.elu(deconv1)

    chan_deconv2 = 32
    W_deconv2 = arch.initialize_weights([3, 3, chan_deconv2, chan_deconv1], _STD_DEV)
    deconv2 = arch.resizeconv2d(act4, W_deconv2, 2)
    deconv2 = arch.instance_normalization(deconv2, chan_deconv2)
    act5 = tf.nn.elu(deconv2)

    chan_conv4 = 3
    W_conv4 = arch.initialize_weights([9, 9, chan_deconv2, chan_conv4])
    conv4 = arch.conv2d(act5, W_conv4, 1)
    conv4 = arch.instance_normalization(conv4, chan_conv4)

    return tf.tanh(conv4) * 150.0 + 127.5


def SingleFrameSR(x):
    #x_pad = arch.spatial_replication_padding(x, 1, utils.tensor_shape(x), (9, 9))
    chan_conv1 = 32
    W_conv1 = arch.initialize_weights([9, 9, 3, chan_conv1], _STD_DEV)
    conv1 = arch.conv2d(x, W_conv1, 1)
    conv1 = arch.instance_normalization(conv1, chan_conv1)
    act1 = tf.nn.elu(conv1)

    chan_conv2 = 64
    W_conv2 = arch.initialize_weights([3, 3, chan_conv1, chan_conv2], _STD_DEV)
    conv2 = arch.conv2d(act1, W_conv2, 2)
    conv2 = arch.instance_normalization(conv2, chan_conv2)
    act2 = tf.nn.elu(conv2)

    chan_conv3 = 128
    W_conv3 = arch.initialize_weights([3, 3, chan_conv2, chan_conv3], _STD_DEV)
    conv3 = arch.conv2d(act2, W_conv3, 2)
    conv3 = arch.instance_normalization(conv3, chan_conv3)
    act3 = tf.nn.elu(conv3)

    W_resid1_1, W_resid1_2 = (
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV))
    resid1 = arch.resid_block(act3, W_resid1_1, W_resid1_2, tf.nn.elu, chan_conv3)
    W_resid2_1, W_resid2_2 = (
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV))
    resid2 = arch.resid_block(resid1, W_resid2_1, W_resid2_2, tf.nn.elu, chan_conv3)
    W_resid3_1, W_resid3_2 = (
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV))
    resid3 = arch.resid_block(resid2, W_resid3_1, W_resid3_2, tf.nn.elu, chan_conv3)
    W_resid4_1, W_resid4_2 = (
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV))
    resid4 = arch.resid_block(resid3, W_resid4_1, W_resid4_2, tf.nn.elu, chan_conv3)
    W_resid5_1, W_resid5_2 = (
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV),
        arch.initialize_weights([3, 3, chan_conv3, chan_conv3], _STD_DEV))
    resid5 = arch.resid_block(resid4, W_resid5_1, W_resid5_2, tf.nn.elu, chan_conv3)

    chan_deconv1 = 64
    W_deconv1 = arch.initialize_weights([3, 3, chan_deconv1, chan_conv3], _STD_DEV)
    deconv1 = arch.resizeconv2d(resid5, W_deconv1, 2)
    deconv1 = arch.instance_normalization(deconv1, chan_deconv1)
    act4 = tf.nn.elu(deconv1)

    chan_deconv2 = 32
    W_deconv2 = arch.initialize_weights([3, 3, chan_deconv2, chan_deconv1], _STD_DEV)
    deconv2 = arch.resizeconv2d(act4, W_deconv2, 2)
    deconv2 = arch.instance_normalization(deconv2, chan_deconv2)
    act5 = tf.nn.elu(deconv2)

    #chan_deconv3 = 16
    #W_deconv3 = arch.initialize_weights([3, 3, chan_deconv3, chan_deconv2], _STD_DEV)
    #deconv3 = arch.resizeconv2d(act5, W_deconv3, 2)
    #deconv3 = arch.instance_normalization(deconv3, chan_deconv3)
    #act7 = tf.nn.elu(deconv3)

    chan_deconv4 = 8
    W_deconv4 = arch.initialize_weights([3, 3, chan_deconv4, chan_deconv3], _STD_DEV)
    deconv4 = arch.resizeconv2d(act6, W_deconv4, 2)
    deconv4 = arch.instance_normalization(deconv4, chan_deconv4)
    act7 = tf.nn.elu(deconv4)

    chan_conv4 = 3
    W_conv4 = arch.initialize_weights([9, 9, chan_deconv4, chan_conv4], _STD_DEV)
    conv4 = arch.conv2d(act7, W_conv4, 1)
    conv4 = arch.instance_normalization(conv4, chan_conv4)

    return tf.tanh(conv4) * 150.0 + 127.5


def Clarity(x):
    pass


def VGG19(path, x, center_data=True, pool_function='MAX'):
    mat = scipy.io.loadmat(path)
    mean_pixel = mat['meta'][0][0][2][0][0][2][0][0]
    net = x
    if center_data:
        net = x - mean_pixel

    pool_funcs = {'AVG': arch.avg_pool, 'MAX': arch.max_pool}

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
            net = arch.conv2d(net, W_conv, 1) + b_conv
        elif layer_type == 'relu':
            net = tf.nn.relu(net)
            layers['relu'+str(pool_num)+'_'+str(relu_num)] = net
            relu_num += 1
        elif layer_type == 'pool':
            net = pool_funcs[pool_function](net, 2)
            pool_num += 1
            relu_num = 1

    return net, layers
