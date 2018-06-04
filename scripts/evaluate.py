#!/usr/bin/python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import argparse

import numpy as np
import PIL.Image as Image
import tensorflow as tf
import pickle

try:
    import scripts.retrain as retrain
    from scripts.count_ops import load_graph
    from scripts.retrain import CACHED_GROUND_TRUTH_VECTORS
except:
    import retrain
    from count_ops import load_graph
    from retrain import CACHED_GROUND_TRUTH_VECTORS

def evaluate_graph(graph_file_name):
    annotation_path = '/Users/azhong/face/face_attrib/lfw_header_lines.p'
    keys_lines = pickle.load(open(annotation_path, 'rb'))
    keys = keys_lines['header']
    lines = keys_lines['lines']
    keys, lines = retrain.prune_data(keys, lines, excluded_keys = ['Male']) # duplicate keys
    labels = keys[2:]
    class_count = len(labels)
    retrain.load_ground_truth_cache(labels, lines)

    with load_graph(graph_file_name).as_default() as graph:
        ground_truth_input = tf.placeholder(
            tf.float32, [None, class_count], name='GroundTruthInput')
        image_buffer_input = graph.get_tensor_by_name('input:0')
        final_tensor = graph.get_tensor_by_name('final_result:0')
        accuracy, _ = retrain.add_evaluation_step(final_tensor, ground_truth_input)
        accuracy_per_class = []
        for i in range(class_count):
            e, _ = retrain.add_evaluation_step_per_class(final_tensor, ground_truth_input, i, labels)
            accuracy_per_class.append(e)

        logits = graph.get_tensor_by_name("final_training_ops/Wx_plus_b/add:0")
        xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels = ground_truth_input,
            logits = logits))

    image_dir = '/Users/azhong/face/data/lfw/lfw_all_funneled'
    testing_percentage = 10
    validation_percentage = 10
    validation_batch_size = 100
    category='testing'

    image_lists = retrain.create_image_lists(
        image_dir, testing_percentage,
        validation_percentage)

    ground_truths = []
    filenames = []

    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(image_lists[label_name][category]):
        image_name = retrain.get_image_path(
            image_lists, label_name, image_index, image_dir, category)
        basename = os.path.basename(image_name).split('.jpg')[0]
        ground_truth = CACHED_GROUND_TRUTH_VECTORS[basename]
        ground_truth = np.array([ground_truth])
        ground_truths.append(ground_truth)
        filenames.append(image_name)

    accuracies = []
    accuracies_per_class = [[] for i in range(class_count)]
    xents = []
    evals = [xent, accuracy]
    for eval in accuracy_per_class:
        evals.append(eval)

    with tf.Session(graph=graph) as sess:
        for filename, ground_truth in zip(filenames, ground_truths):
            image = Image.open(filename).resize((224,224),Image.ANTIALIAS)
            image = np.array(image, dtype=np.float32)[None,...]
            image = (image-127.5)/127.5

            feed_dict={
                image_buffer_input: image,
                ground_truth_input: ground_truth}

            results = sess.run(evals, feed_dict)
            xents.append(results[0])
            accuracies.append(results[1])
            for i in range(class_count):
                accuracies_per_class[i].append(results[i+2])

    return np.mean(xents), np.mean(accuracies), np.mean(accuracies_per_class, axis = 1), labels

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    xent, accuracy, accuracy_per_class, labels = evaluate_graph(*sys.argv[1:])
    print('Cross Entropy: %g' % xent)
    print('Overall Accuracy: %g' % accuracy)
    for i in range(len(labels)):
        print('%s, %g' % (labels[i], accuracy_per_class[i]))
