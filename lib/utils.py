import numpy as np
import os
import fnmatch
import scipy.misc
import shutil
import tensorflow as tf
from functools import reduce
from operator import mul

def read_image(path, mode='RGB', size=None):
    img = scipy.misc.imread(path, mode=mode)
    if size is not None:
        img = scipy.misc.imresize(img, size)
    return img

#def read_video
def write_image(img, path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

def get_output_filepath(output_path, name, suffix='', ext='jpg'):
    return os.path.join(output_path, name+'_'+suffix+'.'+ext)

def gram_matrix(mat):
    mat = mat.reshape((-1, mat.shape[2]))
    return np.matmul(mat.T, mat) / mat.size

def tf_batch_gram_matrix(batch):
    _, height, width, channels = tensor_shape(batch)
    batch = tf.reshape(batch, (-1, height * width, channels))
    batch_T = tf.batch_matrix_transpose(batch)
    return tf.batch_matmul(batch_T, batch) / (height * width * channels)

def tensor_shape(t):
    return tuple(d.value for d in t.get_shape())


def get_train_data_filepaths(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def get_frame_data_filepaths(path,FRAME_SIZE):
    """Using the videos downloaded in `vid_dir`, produce decoded frames in
    `frame_dir`.
    """
    # List the datasets
    imgPaths=list()
    vid_dir=path
    d_sets=[f.path for f in os.scandir(vid_dir) if f.is_dir() ]
    # For each dataset
    for d_set in d_sets:
    # List the classes
        classes=[f.path for f in os.scandir(d_set) if f.is_dir() ]
        # For each class
        for class_ in classes:
            # List the clips
            clips=[f.path for f in os.scandir(class_) ]
            # For each clip, where clip is the path to the clip
            for clip in clips:
                images = [f.path for f in os.scandir(clip)]
                length = len(images)
                # only taking in images in multiple patches of FRAME_SIZE
                for i in range(int(length)//int(FRAME_SIZE) ):
                    for j in range (FRAME_SIZE):
                        imgPaths.append(images[i*FRAME_SIZE+j])
                        # print(images[i*FRAME_SIZE+j])
    return imgPaths
def get_original_image_filepaths(path):
    imgPaths=list()
    for f in os.scandir(path):
        if fnmatch.fnmatch(f.name,'orig_*.jpg'):
            imgPaths.append(f.path)
    return imgPaths

def save_model_with_backup(sess, saver, model_output_path, model_name):
    model_filepath = os.path.join(model_output_path, model_name)
    model_meta_filepath = os.path.join(model_output_path,
            model_name + '.meta')
    if (os.path.isfile(model_filepath)
        and os.path.isfile(model_meta_filepath)):
        shutil.copy2(model_filepath, model_filepath + '.bak')
        shutil.copy2(model_meta_filepath,
                model_meta_filepath + '.bak')

    saver.save(sess, model_filepath)
