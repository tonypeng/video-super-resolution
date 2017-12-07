"""
Created by Tianyi for comparison of SFSR & MFSR
"""
from lib import nets
import numpy as np
import os
import shutil
import tensorflow as tf
from lib import utils
from random import shuffle

FRAME_SIZE = 30
TEST_FRAME_DATASET_PATH='/home/ubuntu/data1/youtubeFramesTest/'
MODEL_PATH='models/trained/MFSR2x/'
CONTENT_IMAGE_SIZE = (256, 256) # (height, width,frames)
DOWNSCALED_CONTENT_IMAGE_SIZE = (128,128) # (height, width)
OUTPUT_PATH = 'runs/Compare/'

saver = tf.train.Saver()

# first get the list of videos to compare
test_data = utils.get_frame_data_filepaths(TEST_FRAME_DATASET_PATH,FRAME_SIZE)
print("Training dataset loaded: " + str(len(train_data)) + " frames.")


# then get the list of models to compare





# run different models on videos
# compare the model outputs with ground truth PSNR/SSIM
# compare only the Y channel after converting to the YCbCr colorspace




# output results
# also output some images every 50?
