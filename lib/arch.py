import tensorflow as tf
from . import utils

def initialize_weights(shape, stddev):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def initialize_biases(shape, value):
    return tf.Variable(tf.constant(value, shape=shape))

def spatial_replication_padding(x, stride, output_shape, filter_shape):
    _, in_height, in_width, chan = utils.tensor_shape(x)
    _, out_height, out_width, chan = output_shape
    filter_height, filter_width = filter_shape


    total_padding_height = (out_height * stride + filter_height - 1) - in_height
    total_padding_width = (out_width * stride + filter_width - 1) - in_width

    padding_top = total_padding_height // 2
    padding_bottom = total_padding_height - padding_top
    padding_left = total_padding_width // 2
    padding_right = total_padding_width - padding_left
    paddings = [padding_top, padding_bottom, padding_left, padding_right]
    while max(paddings) > 0:
        new_paddings = [max(0, p - 1) for p in paddings]
        deltas = [o - n for o, n in zip(paddings, new_paddings)]
        step_paddings = [[0, 0], [deltas[0], deltas[1]], [deltas[2], deltas[3]], [0, 0]]
        x = tf.pad(x, step_paddings, mode='SYMMETRIC')
        paddings = new_paddings


def conv2d(x, W, stride, border_mode='SAME'):
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], border_mode)

def deconv2d(x, W, stride, output_shape, border_mode='SAME'):
    return tf.nn.conv2d_transpose(x, W, output_shape, [1, stride, stride, 1], padding=border_mode)

def resizeconv2d(x, W, stride):
    x_size = x.get_shape().as_list()
    resized_height, resized_width = x_size[1] * stride * stride, x_size[2] * stride * stride
    x_resized = tf.image.resize_images(x, (resized_height, resized_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    W_T = tf.transpose(W, perm=[0, 1, 3, 2])
    return conv2d(x_resized, W_T, stride, border_mode='SAME')

def avg_pool(x, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)

def max_pool(x, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)

def resid_block(x, W1, W2, activation, channels):
    conv1 = conv2d(x, W1, 1)
    conv1 = instance_normalization(conv1, channels)
    act1 = activation(conv1)

    conv2 = conv2d(act1, W2, 1)
    conv2 = activation(conv2)

    return conv2 + x

def instance_normalization(x, channels):
    instance_mean, instance_var = tf.nn.moments(x, [1, 2], keep_dims=True)
    epsilon = 1e-5
    x_hat = (x - instance_mean) / tf.sqrt(instance_var + epsilon)
    scale = tf.Variable(tf.ones([channels]))
    offset = tf.Variable(tf.zeros([channels]))
    return scale * x_hat + offset
