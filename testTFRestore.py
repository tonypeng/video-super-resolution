from lib import nets
import numpy as np
import os
import shutil
import tensorflow as tf
from lib import utils
from random import shuffle

CONTENT_WEIGHT = 1 
STYLE_WEIGHT = 0
DENOISE_WEIGHT = 5
LEARNING_RATE = 1e-3
EPOCHS = 2

DEVICE = '/gpu:0'
MODEL_OUTPUT_PATH = 'models/trained/SFSR'
META_PATH='models/trained/SFSR/model.meta'
MODEL_PATH='models/trained/SFSR/'
CHECKPOINT_PATH='models/trained/SFSR/checkpoint'
MODEL_NAME='SFSR2x'
TRAIN_DATASET_PATH = '/home/ubuntu/data1/coco/images/train2014'
TEST_FRAME_DATASET_PATH='/home/ubuntu/data1/youtubeFramesTest/'
VGG_MODEL_PATH = 'models/vgg/imagenet-vgg-verydeep-19.mat'
#STYLE_IMAGE_PATH = 'runs/WhiteLine/style.jpg'
CONTENT_IMAGE_SIZE = (320,320) # (height, with)
DOWNSCALED_CONTENT_IMAGE_SIZE = (160,160) # (height, width)
#STYLE_SCALE = 1.0
# make sure all input are taken in 30 a batch
MINI_BATCH_SIZE = 16
#VALIDATION_IMAGE_PATH = 'runs/WhiteLine/content.jpg'
OUTPUT_PATH = 'runs/Compare/'
PREVIEW_ITERATIONS = 50
CHECKPOINT_ITERATIONS = 500
CONTENT_LAYER = 'relu4_2'
TEST_SIZE=1000
# layer: w_l

 # batch shape is (batch, height, width, channels)
batch_shape = (MINI_BATCH_SIZE, ) + CONTENT_IMAGE_SIZE + (3, )
#style_image = utils.read_image(STYLE_IMAGE_PATH,
#        size=tuple(int(d * STYLE_SCALE) for d in CONTENT_IMAGE_SIZE))

train_data = utils.get_frame_data_filepaths(TEST_FRAME_DATASET_PATH,MINI_BATCH_SIZE)
print("Training dataset loaded: " + str(len(train_data)) + " images.")

#validation_image = utils.read_image(VALIDATION_IMAGE_PATH, size=CONTENT_IMAGE_SIZE)

def evaluate_stylzr_output(t, feed_dict=None):
    return t.eval(feed_dict=feed_dict)

output_evaluator = evaluate_stylzr_output

g = tf.Graph()
with g.as_default(), g.device(DEVICE),tf.Session(
    config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    content_batch = tf.placeholder(tf.float32, shape=batch_shape,
            name="input_content_batch")

    # Construct transfer network
    print("3. Constructing style transfer network...")
    # transfer_net = nets.gatys(gatys_content_image.shape)
    transfer_net = nets.SingleFrameSR(tf.image.resize_images(content_batch, DOWNSCALED_CONTENT_IMAGE_SIZE))

    sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver()
    saver = tf.train.import_meta_graph(META_PATH)
    #reader = tf.train.NewCheckpointReader(CHECKPOINT_PATH)
    #print('reder:\n', reader)
    input("Press Enter to continue...")
    saver.restore(sess, (MODEL_PATH))

