import json
import os
import time

import utils
import frame_level_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib

FLAGS = flags.FLAGS

flags.DEFINE_string("train_dir", "../data/youtube8M/",
                      "The directory to save the model files in.")
flags.DEFINE_string(
      "train_data_pattern", "../data/youtube8M/train*.tfrecord",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
flags.DEFINE_string("feature_names", "rgb", "Name of the feature "
                      "to use for training.")
flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
flags.DEFINE_bool(
      "frame_features", True,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
flags.DEFINE_integer("batch_size", 128,
                       "How many examples to process per batch for training.")
flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
flags.DEFINE_float(
      "regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")
flags.DEFINE_float("learning_rate_decay", 0.95,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")
flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")
flags.DEFINE_integer("num_epochs", 1,
                       "How many passes to make over the dataset before "
                       "halting training.")
flags.DEFINE_integer("max_steps", None,
                       "The maximum number of iterations of the training loop.")
flags.DEFINE_integer("export_model_steps", 1000,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")

  # Other flags.
flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

print("Using batch size of " + str(FLAGS.batch_size) + " for training.")
files = gfile.Glob(FLAGS.train_data_pattern)
print("Number of training files:" +str(len(files)))
filename_queue = tf.train.string_input_producer(files, num_epochs=FLAGS.num_epochs, shuffle=False)

feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)
training_data = reader.prepare_reader(filename_queue)

print(training_data)
batch_video_ids, batch_video_matrix, batch_labels, batch_frames = training_data

i = tf.Print(batch_frames,[batch_frames])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    m_eval = i.eval(session=sess,feed_dict={batch_frames:batch_frames})



