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

saver = tf.train.Saver()

# first get the list of videos to compare



# then get the list of models to compare





# run different models on videos
# compare the model outputs with ground truth PSNR/SSIM
# compare only the Y channel after converting to the YCbCr colorspace




# output results
# also output some images every 50?
