from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import random

import os
import json
import common

parser = argparse.ArgumentParser()
parser.add_argument("--input-path")
parser.add_argument("--model-dir-path")

args = parser.parse_args()

training_data = common.read_data(args.input_path)
dictionary, reverse_dictionary = common.build_dataset(training_data, dict())
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
n_input = common.get_n_input()
pred, x, y = common.init_model(dictionary, n_input)

saver = tf.train.Saver()

# Loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar('loss', cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar('lr', learning_rate)

# Model evaluation
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_pred'):
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# training_iters = 50000
training_iters = 5000
display_step = 10

#Summary Uncomment for TF stats see bellow
# merged = tf.summary.merge_all()
# test_writer = tf.summary.FileWriter('/tmp/logs/test')

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    
    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)

        symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            # Uncomment for TF stats see above
            # summary = session.run(merged, feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            # test_writer.add_summary(summary, step)

            print("Iter= " + str(step + 1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100 * acc_total / display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
        step += 1
        offset += (n_input + 1)

    saver.save(session, os.path.join(args.model_dir_path, "model"))
    with open(os.path.join(args.model_dir_path, "dictionary.json"), "w") as dict_file:
        dict_file.write(json.dumps(dictionary))
    print("Saved training result to {}".format(args.model_dir_path))
