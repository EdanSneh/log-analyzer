from __future__ import print_function
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
import numpy as np
from io import StringIO

import argparse
import common
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input-path")
parser.add_argument("--model-dir-path")

args = parser.parse_args()

with open(os.path.join(args.model_dir_path, "dictionary.json"), "r") as dict_file:
    dict = json.load(dict_file)

data = common.read_data(args.input_path)
dictionary, reverse_dictionary = common.build_dataset(data, dict)
n_input = common.get_n_input()

vocab_size = len(dictionary)

#gRPC
server = '35.185.218.27:9000'
host, port = server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'log-server'
request.model_spec.signature_name = 'log_guess'

for sentences in np.array_split(data, len(data) / n_input):
    if len(sentences) == n_input:
        symbols_in_keys = [dictionary[str(sentences[i])] for i in range(len(sentences))]
        for i in range(1):
            keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
            ###client stuff
            request.inputs['actual_logs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(keys.astype(dtype=np.float32), shape=[n_input,1]))
            result_future = stub.Predict(request, 30.0)
            output = np.array(result_future.outputs['predicted_logs'].float_val)
            print(output)
            # logs = "%s %s" % (sentences, reverse_dictionary[onehot_pred_index])
            # symbols_in_keys = symbols_in_keys[1:]
            # symbols_in_keys.append(onehot_pred_index)

