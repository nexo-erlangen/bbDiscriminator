#!/usr/bin/env python
"""
Copyright (c) 2018, by the Authors: Amir H. Abdi
This script is freely available under the MIT Public License.
Please see the License file in the root for details.

The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""

import tensorflow as tf
import keras
from keras import backend as K
from absl import app
from absl import flags
import os

K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('input_model_weights', None, 'Path to the input model '
                                              'architecture in json format.')
flags.DEFINE_string('output_name', None, 'Path where the converted model will '
                                          'be stored.')

flags.mark_flag_as_required('input_model_weights')
flags.mark_flag_as_required('output_name')


def save_model_with_weights(input_model_weights, input_model_temp):
    if not os.path.isfile(input_model_weights):
        raise FileNotFoundError('Model file `{}` does not exist.'.format(input_model_weights))
    try:
        try:
            model = keras.models.load_model(os.path.join(os.path.dirname(input_model_weights), 'model-000.hdf5'))
        except:
            model = keras.models.load_model(os.path.join(os.path.dirname(input_model_weights), 'model-initial.hdf5'))
        model.load_weights(input_model_weights, by_name=False)
        model.save(input_model_temp)
    except:
        if os.path.isfile(input_model_temp): os.remove(input_model_temp)
        raise ValueError('something wrong when reading/writing keras model')
    return

def main(args):
    input_model_path = os.path.join(os.path.dirname(FLAGS.input_model_weights), 'model-temp-XXXXXX.hdf5')
    output_model_path = os.path.join(os.path.dirname(FLAGS.input_model_weights), FLAGS.output_name+'.pb')
    save_model_with_weights(FLAGS.input_model_weights, input_model_path)
    try:
        exe = '/home/hpc/capm/sn0515/keras_to_tensorflow/keras_to_tensorflow.py'
        cmd = 'python %s --input_model=%s --output_model=%s'%(exe, input_model_path, output_model_path)
        print cmd
        os.system(cmd)
    except:
        if os.path.isfile(output_model_path): os.remove(output_model_path)
        raise ValueError('converting Keras model to TF model failed')
    finally:
        if os.path.isfile(input_model_path): os.remove(input_model_path)

if __name__ == "__main__":
    app.run(main)
