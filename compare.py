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

