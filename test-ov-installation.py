# Run this in IPython

import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import decode_predictions
import numpy as np
import shutil

model = InceptionV3(weights='imagenet')
model_fname = './inceptionv3/model.h5'

# Save the h5 file to path specified.
if os.path.isdir('./inceptionv3'):
    print ('./inceptionv3', "exists already. Deleting the folder")
    shutil.rmtree('./inceptionv3')
os.mkdir('./inceptionv3')
model.save(model_fname)

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model

tf.keras.backend.clear_session()

save_pb_dir = './inceptionv3/frozen_model/'
model_fname = './inceptionv3/model.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

tf.keras.backend.set_learning_phase(0) 
model = load_model(model_fname)
session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

!mo_tf.py \
      --input_model './inceptionv3/frozen_model/frozen_model.pb' \
      --input_shape=[1,299,299,3] \
      --data_type FP32  \
      --output_dir './inceptionv3/IR_models/FP32'  \
      --model_name inceptionv3

from openvino.inference_engine import IENetwork, IECore

!python3 /opt/intel/openvino/deployment_tools/tools/benchmark_tool/benchmark_app.py \
-m './inceptionv3/IR_models/FP32/inceptionv3.xml' \
-nireq 1 -nstreams 1 -t 10
