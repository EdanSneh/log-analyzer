from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

import os
import json
import common

parser = argparse.ArgumentParser()
parser.add_argument("--input-path")
parser.add_argument("--model-dir-path")

args = parser.parse_args()

with open(os.path.join("dictionary", "dictionary.json"), "r") as dict_file:
    dict = json.load(dict_file)

data = common.read_data(args.input_path)
dictionary, reverse_dictionary = common.build_dataset(data, dict)

vocab_size = len(dictionary)

n_input = common.get_n_input()
pred, x, y = common.init_model(dictionary, n_input)

saver = tf.train.Saver()
with tf.Session() as session:
    saver.restore(session, args.model_dir_path+"/model")
    print("Model restored from {}".format(args.model_dir_path))
    for words in np.array_split(data, len(data) / n_input):
        if len(words) == n_input:
            sentence = ""
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                #shows model accuracy
                print(onehot_pred)
                #TODO see if given log matches any of the highest rated logs to see if it is common
            # print(sentence)
