from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

import os
import json
import common

parser = argparse.ArgumentParser()
parser.add_argument("--input-path")
parser.add_argument("--model-output-path")
parser.add_argument("--version")
args = parser.parse_args()


with open(os.path.join(args.input_path, "dictionary.json"), "r") as dict_file:
    dict = json.load(dict_file)

#initialize variables and model
n_input = common.get_n_input()
pred, x, y = common.init_model(dict, n_input)

tensor_info_input = tf.saved_model.utils.build_tensor_info(x)
tensor_info_output = tf.saved_model.utils.build_tensor_info(pred)

#model output

#restore model
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, os.path.join(args.input_path, "model"))

#specify path to build model to based on version
export_path_base = args.model_output_path
export_path = os.path.join(
    export_path_base,
    args.version)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

#signature (i/o)
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'actual_logs': tensor_info_input},
        outputs={'predicted_logs': tensor_info_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'log_guess':
            prediction_signature,
    })

#export the model
builder.save(as_text=True)

