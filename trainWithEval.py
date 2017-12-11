"""
Copyright 2016-present Tony Peng

Implementation of the papers "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
by Justin Johnson, Alexandre Alahi, and Li Fei-Fei and "A Neural Algorithm of Artistic Style"
by Leon Gatys, Alexander S Ecker, and Matthias Bethge.
"""

from lib import nets
from lib import psnr
from lib import ssim
import numpy as np
import os
import shutil
import tensorflow as tf
from lib import utils
from random import shuffle
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
CONTENT_WEIGHT = 1 
STYLE_WEIGHT = 0
DENOISE_WEIGHT = 5
LEARNING_RATE = 1e-3
EPOCHS = 2

TEST_FRAME_DATASET_PATH='/home/ubuntu/data1/youtubeFramesTest/'
FRAME_SIZE = 30
CHECKPOINT_ITERATIONS = 500
TEST_SIZE = 50
TEST_OUTPUT_PATH = 'runs/SFSR2xWithEval/test'
TEST_ITERATIONS = 100

DEVICE = '/gpu:0'
MODEL_OUTPUT_PATH = 'models/trained/SFSR2xWithEval'
MODEL_NAME = 'model'
TRAIN_DATASET_PATH = '/home/ubuntu/data1/coco/images/train2014'
VGG_MODEL_PATH = 'models/vgg/imagenet-vgg-verydeep-19.mat'
#STYLE_IMAGE_PATH = 'runs/WhiteLine/style.jpg'
CONTENT_IMAGE_SIZE =(256,256) # (height, width)
DOWNSCALED_CONTENT_IMAGE_SIZE = (128,128) # (height, width)
#STYLE_SCALE = 1.0
MINI_BATCH_SIZE = 30
#VALIDATION_IMAGE_PATH = 'runs/WhiteLine/content.jpg'
OUTPUT_PATH = 'runs/SFSR2xWithEval/train'
PREVIEW_ITERATIONS = 50
CONTENT_LAYER = 'relu4_2'
# layer: w_l
STYLE_LAYERS = {
    'relu1_1': 0.2,
    'relu2_1': 0.2,
    'relu3_1': 0.2,
    'relu4_1': 0.2,
    'relu5_1': 0.2,
}

 # batch shape is (batch, height, width, channels)
batch_shape = (MINI_BATCH_SIZE, ) + CONTENT_IMAGE_SIZE + (3, )
#style_image = utils.read_image(STYLE_IMAGE_PATH,
#        size=tuple(int(d * STYLE_SCALE) for d in CONTENT_IMAGE_SIZE))

train_data = utils.get_train_data_filepaths(TRAIN_DATASET_PATH)
print("Training dataset loaded: " + str(len(train_data)) + " images.")

#validation_image = utils.read_image(VALIDATION_IMAGE_PATH, size=CONTENT_IMAGE_SIZE)

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

    content_batch = tf.placeholder(tf.float32, shape=batch_shape,
            name="input_content_batch")

    # Create content target
    print("2. Creating content target...")
    content_net, content_layers = nets.VGG19(VGG_MODEL_PATH, content_batch)
    content_target = content_layers[CONTENT_LAYER]

    # Construct transfer network
    print("3. Constructing style transfer network...")
    # transfer_net = nets.gatys(gatys_content_image.shape)
    transfer_net = nets.SingleFrameSR(tf.image.resize_images(content_batch, DOWNSCALED_CONTENT_IMAGE_SIZE))

    # Set up losses
    print("4. Constructing loss network...")
    loss_network, loss_layers = nets.VGG19(VGG_MODEL_PATH, transfer_net)
    print("5. Creating losses...")
    loss_content = (tf.nn.l2_loss(loss_layers[CONTENT_LAYER] - content_target)
            / tf.to_float(tf.size(content_target)))

    loss_tv = (
        (tf.nn.l2_loss(transfer_net[:, 1:, :, :] - transfer_net[:, :batch_shape[1]-1, :, :]) / tf.to_float(tf.size(transfer_net[0, 1:, :, :]))
            + tf.nn.l2_loss(transfer_net[:, :, 1:, :] - transfer_net[:, :, :batch_shape[2]-1, :]) / tf.to_float(tf.size(transfer_net[0, :, 1:, :])))
        / MINI_BATCH_SIZE
    )

    loss = CONTENT_WEIGHT * loss_content + DENOISE_WEIGHT * loss_tv

    # Optimize
    print("6. Optimizing...")
    optimize = (tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
                    .minimize(loss))

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('models/trained/SFSR/model.meta')
    #saver.restore(sess,'models/trained/SFSR/model')
    global_it = 0
    LogFilePath = os.path.join(MODEL_OUTPUT_PATH+'/score.log')
    LogFilePath = 'SFSR2xWithEval.log'
    for n in range(EPOCHS):
        shuffle(train_data)
        for s in range(0, len(train_data), MINI_BATCH_SIZE):
            global_it_num = global_it + 1
            batch = np.array([utils.read_image(f, size=CONTENT_IMAGE_SIZE)
                    for f in train_data[s:s+MINI_BATCH_SIZE]])
            if len(batch) < MINI_BATCH_SIZE:
                print(
                    "Skipping mini-batch because there are not enough samples.")
                continue

            _, curr_loss = sess.run([optimize, loss],
                                    feed_dict={content_batch: batch})
            print("Iteration "+str(global_it_num)+": Loss="+str(curr_loss))

            if global_it_num % PREVIEW_ITERATIONS == 0:
                curr_styled_images = output_evaluator(transfer_net,
                        feed_dict={content_batch: batch})
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
            if global_it_num % TEST_ITERATIONS ==0:
                print("Evaluating...")
                # read all test files
                test_data = utils.get_frame_data_filepaths(TEST_FRAME_DATASET_PATH,FRAME_SIZE)
                PSNR = np.zeros(TEST_SIZE)
                SSIM = np.zeros(TEST_SIZE)
                s = 0
                for k in range(TEST_SIZE):
                # actual test size = TEST_SIE x FRAME_SIZE
                    batchTest = np.array([utils.read_image(f, size=CONTENT_IMAGE_SIZE)
                        for f in test_data[s:s+FRAME_SIZE]])
                    styleTest = output_evaluator(transfer_net,feed_dict={content_batch: batch})
                    tmpPSNR = np.zeros(FRAME_SIZE)
                    tmpSSIM = np.zeros(FRAME_SIZE)
                    for t in range(FRAME_SIZE):
                        tmpPSNR[t] = psnr.psnr(batchTest[t],styleTest[t])
                        tmpSSIM[t] = ssim.ssim_exact(batchTest[t],styleTest[t])
                    PSNR[k] = np.mean(tmpPSNR)
                    SSIM[k] = np.mean(tmpSSIM)
                PSNRMean = np.mean(PSNR)
                SSIMMean = np.mean(SSIM)
                with open(LogFilePath,'a') as f:
                    f.write('Iteration {0}, PSNR: {1}, SSIM: {2}\n'.
                        format(global_it_num,PSNRMean,SSIMMean))
                print('{0} evluation done.'.format(global_it_num))
                styled_output_path = utils.get_output_filepath(TEST_OUTPUT_PATH,
                        'styled', str(global_it_num))
                orig_output_path = utils.get_output_filepath(TEST_OUTPUT_PATH,
                        'orig', str(global_it_num))
                utils.write_image(styleTest[0], styled_output_path)
                utils.write_image(batchTest[0], orig_output_path)


    utils.save_model_with_backup(sess, saver, MODEL_OUTPUT_PATH, MODEL_NAME)
    f.close()
print("7: Profit!")
