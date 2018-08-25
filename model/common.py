from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import re

# number of units in RNN cell
n_hidden = 512

'''
Args:
    x: tf input
    weights: weights tf variable
    biases: biases tf variable
    n_input: batch size
Returns:
    tf.matmul: RNN definition

returns an RNN
'''
def RNN(x, weights, biases, n_input):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

'''
Args:
    fname: a files directoy
Returns:
    content: np array of words in file

Seperates words in file to be parsed later
'''
def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    totaloutput=[]
    for line in content:
        # output = re.search(r'^(time)="(.*?)T', line).groups()
        input = re.search(r'Z" (.*)', line).groups()
        totaloutput = totaloutput + [input[0]]
    totaloutput = np.array(totaloutput)
    return totaloutput

'''
Args:
    words: takes in parsed words
    dictionary: a dictionary of all the known words so far
Returns:
    dictionary: updated dictionary of all the known words so far
    reverse_dictionary: reverse of dictionary

Updates dictionary with new words learned in log and returns forwards and backwards dictionary
'''
def build_dataset(words, dictionary):
    count = collections.Counter(words).most_common()
    words_set = set()
    for word, _ in count:
        words_set.add(word)
    new_words = words_set - dictionary.keys()
    for word in new_words:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

'''
Args:
    dictionary: a dictionary of all the known words so far
    n_input: batch size
Returns:
    RNN: RNN LSTM model
    x: tf inputs
    y: tf outputs

Initializes variables and RNN model
'''
def init_model(dictionary, n_input):
    vocab_size = len(dictionary)

    # tf Graph input
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, vocab_size])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }
    return RNN(x, weights, biases, n_input), x, y

'''get package globalized n_input'''
def get_n_input():
    return 3



'''
Give an ML summary
'''
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)