    # -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:15:02 2018

@author: SKT
"""


import tensorflow as tf
import os
from tensorflow.python.platform import gfile
import numpy as np
import utils
from scipy import misc
import matplotlib.pyplot as plt

def load_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def euclidean_distance(vects):
    x, y = vects
    return np.sqrt(np.sum(np.square(x - y)))

NAME = 'test'
DATA_PATH = 'data/' + NAME + '/'
COMPARE_PATH = 'compare/' + NAME + '/'

def make_compare_data():

    emb_dict = load()
    img_dict = {}

    for folder_name in os.listdir(DATA_PATH):

        if folder_name in emb_dict.keys():
            continue

        print(folder_name)

        folder_path = DATA_PATH + folder_name + '/'
        img_paths = np.random.choice([file_name for file_name in os.listdir(folder_path)], 3, replace=False)

#        print(img_paths)

        img_dict[folder_name] = []

        for i,img_path in enumerate(img_paths):
            img_path = folder_path + img_path
            face = utils.get_face(img_path)[0]
            face = prewhiten(face)
            img_dict[folder_name].append(face)
            misc.imsave(COMPARE_PATH + folder_name + '_' + str(i+1) + '.png', face)

    new_emb_dict = get_embedding(img_dict, make_data = True)
    emb_dict.update(new_emb_dict)
    np.save(COMPARE_PATH + NAME + '.npy', emb_dict)


def load():
    try:
        emb_dict = np.load(COMPARE_PATH + NAME + '.npy')
        emb_dict = emb_dict[()]
    except:
        emb_dict = {}

    return emb_dict

def get_embedding(data, make_data = False):

    return_data = None

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.09)

    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            #load model
            load_model('facenet.pb')

            if make_data:

                img_dict = data
                emb_dict = {}

                for key,imgs in img_dict.items():

                    images = np.array(imgs)

                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    feed_dict = { images_placeholder: images , phase_train_placeholder:False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    emb_dict[key] = emb

                return_data = emb_dict

            else:

                img = data

                face = utils.get_face(img, path=False)

                if len(face) == 0:
                    return

                else:
                    face = face[0]

                face = prewhiten(face)

                images = np.array([face])

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                feed_dict = { images_placeholder: images , phase_train_placeholder:False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                return_data = emb[0]

        return return_data

def predict(img):

    scores = {}
    best_match = ''
    best_score = 100

    img_emb = get_embedding(img)

    if img_emb is None:
        return

    emb_dict = load()

    for name,embs in emb_dict.items():
        score = 0
        for emb in embs:
            score += euclidean_distance((img_emb, emb))

        score = score/len(embs)

        scores[name] = score

        if score < best_score:
            best_score = score
            best_match = name

    return scores, best_match, best_score