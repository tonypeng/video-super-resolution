"""
Copyright 2016-present Tony Peng
RNN structure built by Tianyi Zeng 2017

Implementation of the papers "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
by Justin Johnson, Alexandre Alahi, and Li Fei-Fei and "A Neural Algorithm of Artistic Style"
by Leon Gatys, Alexander S Ecker, and Matthias Bethge.
"""

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
MODEL_OUTPUT_PATH = 'models/trained/MFSR'
MODEL_NAME = 'model'
TRAIN_DATASET_PATH = '/home/ubuntu/dataset/trainvideo'
FRAME_DATASET_PATH = '/home/ubuntu/data1/youtubeFrames/'
VGG_MODEL_PATH = 'models/vgg/imagenet-vgg-verydeep-19.mat'
#STYLE_IMAGE_PATH = 'runs/WhiteLine/style.jpg'
FRAME_SIZE = 30
CONTENT_IMAGE_SIZE = (256, 256) # (height, width,frames)
DOWNSCALED_CONTENT_IMAGE_SIZE = (128,128) # (height, width)
#STYLE_SCALE = 1.0
OUTPUT_PATH = 'runs/MFSR/2x'

PREVIEW_ITERATIONS = 50
CHECKPOINT_ITERATIONS = 500
CONTENT_LAYER = 'relu4_2'
# layer: w_l
STYLE_LAYERS = {
    'relu1_1': 0.2,
    'relu2_1': 0.2,
    'relu3_1': 0.2,
    'relu4_1': 0.2,
    'relu5_1': 0.2,
}

 # batch shape is (height, width, channels, frames)
batch_shape = (FRAME_SIZE, ) + CONTENT_IMAGE_SIZE + (3, )
#style_image = utils.read_image(STYLE_IMAGE_PATH,
       # size=tuple(int(d * STYLE_SCALE) for d in CONTENT_VIDEO_SIZE))

# probs get all the starting file names of each 30 frames 
train_data = utils.get_frame_data_filepaths(FRAME_DATASET_PATH,FRAME_SIZE)
print("Training dataset loaded: " + str(len(train_data)) + " frames.")


def evaluate_stylzr_output(t, feed_dict=None):
    return t.eval(feed_dict=feed_dict)

output_evaluator = evaluate_stylzr_output

# Overrides for Gatys style transfer
# CONTENT_IMAGE_PATH = 'KillianCourt.jpg'
# gatys_content_image = utils.read_image(CONTENT_IMAGE_PATH)
# CONTENT_IMAGE_SIZE = gatys_content_image.shape[:2]
# train_data = np.array([CONTENT_IMAGE_PATH])
# batch_shape = (1, ) + gatys_content_image.shape
# style_image = utils.read_image(STYLE_IMAGE_PATH,
#         size=tuple(gatys_content_image.shape[:2]))
#
# def evaluate_gatys_output(t, **kwargs):
#     return np.clip(t.eval(), 0, 255).astype(np.uint8)
#
# output_evaluator = evaluate_gatys_output
# End overrides for Gatys style transfer

g = tf.Graph()
with g.as_default(), g.device(DEVICE), tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    #style_input = tf.placeholder(tf.float32, (1,) + style_image.shape)
    prev_prediction_batch = tf.placeholder(tf.float32, shape=batch_shape,
            name="prev_prediction_content_batch")
    post_prediction_batch = tf.placeholder(tf.float32, shape=batch_shape,
            name="post_prediction_content_batch")

    content_batch = tf.placeholder(tf.float32, shape=batch_shape,
            name="input_content_batch")
    # Pre-compute style gram matrices
    #print("1. Pre-computing style Gram matrices...")
    #style_net, style_layers = nets.vgg(VGG_MODEL_PATH, style_input)
    #grams = {}
    #for layer, _ in STYLE_LAYERS.items():
        #feature_maps = style_layers[layer].eval(
         #       feed_dict={style_input: np.array([style_image])})
    #   grams[layer] = utils.gram_matrix(feature_maps[0])
    # Clean up


    # Create content target
    print("2. Creating content target...")
    content_net, content_layers = nets.VGG19(VGG_MODEL_PATH, content_batch)
    content_target = content_layers[CONTENT_LAYER]

    # Construct transfer network
    print("3. Constructing style transfer network...")
    # transfer_net = nets.gatys(gatys_content_image.shape)
    transfer_net = nets.MFSR(tf.image.resize_images(prev_prediction_batch,
        DOWNSCALED_CONTENT_IMAGE_SIZE),
        tf.image.resize_images(content_batch,DOWNSCALED_CONTENT_IMAGE_SIZE),
        tf.image.resize_images(post_prediction_batch,
        DOWNSCALED_CONTENT_IMAGE_SIZE)
        )

    # Set up losses
    print("4. Constructing loss network...")
    loss_network, loss_layers = nets.VGG19(VGG_MODEL_PATH, transfer_net)
    print("5. Creating losses...")
    print()
    # simple loss
    loss_content = (tf.nn.l2_loss(loss_layers[CONTENT_LAYER] - content_target)
            / tf.to_float(tf.size(content_target)))

    #loss_style = 0
    #for layer, w_l in STYLE_LAYERS.items():
        #feature_maps = loss_layers[layer]
        #gram = utils.tf_batch_gram_matrix(feature_maps)
        #gram_target = grams[layer]
        #loss_style += w_l * tf.nn.l2_loss(gram_target - gram) / (gram_target.size * MINI_BATCH_SIZE)

    loss_tv = (
        (tf.nn.l2_loss(transfer_net[:, 1:, :, :] - transfer_net[:, :batch_shape[1]-1, :, :]) / tf.to_float(tf.size(transfer_net[0, 1:, :, :]))
            + tf.nn.l2_loss(transfer_net[:, :, 1:, :] - transfer_net[:, :, :batch_shape[2]-1, :]) / tf.to_float(tf.size(transfer_net[0, :, 1:, :])))
        / FRAME_SIZE
    )

    loss = CONTENT_WEIGHT * loss_content  + DENOISE_WEIGHT * loss_tv

    # Optimize
    print("6. Optimizing...")
    optimize = (tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                    .minimize(loss))

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    
    global_it = 0
    for n in range(EPOCHS):
        shuffle(train_data)
        for s in range(0, len(train_data), FRAME_SIZE):
            global_it_num = global_it + 1
            batch = np.array([utils.read_image(f, size=CONTENT_IMAGE_SIZE)
                    for f in train_data[s:s+FRAME_SIZE]])
            if len(batch) < FRAME_SIZE:
                print(
                    "Skipping mini-batch because there are not enough samples.")
                continue
           
            prevBatchPrediction = np.array(batch)
            postBatchPrediction = np.array(batch)
            
            batchPrediction = output_evaluator(transfer_net,
                feed_dict={prev_prediction_batch: prevBatchPrediction, 
                content_batch: batch,post_prediction_batch:postBatchPrediction })
            # create the prev_prediction_batch & post_prediction_batch
            for i in range(FRAME_SIZE):
                if i ==0:
                    prevBatchPrediction[i]=batchPrediction[i]
                    postBatchPrediction[i]=batchPrediction[i+1] 
                elif i == FRAME_SIZE-1:
                    prevBatchPrediction[i]=batchPrediction[i-1]
                    postBatchPrediction[i]=batchPrediction[i] 
                else:
                    prevBatchPrediction[i]=batchPrediction[i-1]
                    postBatchPrediction[i]=batchPrediction[i+1] 
            _, curr_loss = sess.run([optimize, loss],
                                    feed_dict={prev_prediction_batch: prevBatchPrediction, 
                                    content_batch: batch,
                                    post_prediction_batch:postBatchPrediction })

            print("Iteration "+str(global_it_num)+": Loss="+str(curr_loss))
            if global_it_num % PREVIEW_ITERATIONS == 0:

                curr_styled_images = output_evaluator(transfer_net,
                    feed_dict={prev_prediction_batch: prevBatchPrediction, 
                    content_batch: batch,post_prediction_batch:postBatchPrediction })
                # take the first images
                curr_styled_image = curr_styled_images[0]
                curr_orig_image = batch[0]
                styled_output_path = utils.get_output_filepath(OUTPUT_PATH,
                        'styled', str(global_it_num))
                orig_output_path = utils.get_output_filepath(OUTPUT_PATH,
                        'orig', str(global_it_num))
                utils.write_image(curr_styled_image, styled_output_path)
                utils.write_image(curr_orig_image, orig_output_path)

            if global_it_num % CHECKPOINT_ITERATIONS == 0:
                utils.save_model_with_backup(sess, saver, MODEL_OUTPUT_PATH, MODEL_NAME)
            global_it += 1

    utils.save_model_with_backup(sess, saver, MODEL_OUTPUT_PATH, MODEL_NAME)
print("7: Profit!")

#>>>>>>> f7cc0ea291c0db652929313cd7570258b37e58e2
