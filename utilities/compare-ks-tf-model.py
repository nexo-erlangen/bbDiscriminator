import json, argparse, time

import tensorflow as tf
import h5py
import numpy as np
import keras as ks
from datetime import datetime
# from load import load_graph

# from flask import Flask, request
# from flask_cors import CORS

##################################################
# API part
# ##################################################
# app = Flask(__name__)
# cors = CORS(app)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    print [n.name + '=>' + n.op for n in graph_def.node if n.op in ('Softmax', 'Placeholder')]
    return graph

def generate_batches_from_files(files, batchsize, wires=None, class_type=None, f_size=None, select_dict={}, yield_mc_info=0):
    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variable need to be list/np.array/str/dict[dict]')

    if wires == 'U':    wireindex = [0, 2]
    elif wires == 'V':  wireindex = [1, 3]
    elif wires == 'small':  wireindex = slice(2)
    elif wires in ['UV', 'U+V']: wireindex= slice(4)
    else: raise ValueError('passed wire specifier need to be U/V/UV/small. Not: %s'%(wires))

    eventInfo = {}
    while 1:
        for filename in files:
            f = h5py.File(str(filename), "r")

            for key in f.keys():
                if key in ['wfs']: continue
                eventInfo[key] = np.asarray(f[key])

            if not yield_mc_info in [-1,2]:
                xs = np.asarray(f['wfs'])[:, wireindex]
                # print xs.shape

                #TODO these 2 lines for baseline U-only 2x(Bx350x38x1)
                xs = np.swapaxes(xs, 0, 1)
                xs = np.swapaxes(xs, 2, 3)

                #TODO these 2 lines for UV (Bx350x38x4)
                # xs = np.swapaxes(xs, 1, 3)
                # xs = np.squeeze(xs)

                #TODO these 2 lines for U-only (Bx350x76x1)
                # xs = np.reshape(xs, (xs.shape[0], 76, 350, -1))
                # xs = np.swapaxes(xs, 1, 2)

                # print xs.shape
                # exit()

            lst = np.arange(0, 4000, batchsize)
            for i in lst:
                batch = sorted(lst[i: i + batchsize])

                if not yield_mc_info in [-1,2]:
                    #TODO this line for baseline U-only 2x(Bx350x38x1)
                    xs_i = xs[:, batch]

                if   yield_mc_info == 0:
                    yield list(xs_i)
            f.close()

##################################################
# END API part
##################################################

if __name__ == "__main__":


    # import keras as ks
    #
    # path = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181030-1854/'
    # path = '/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/181217-1612/'
    # model = ks.models.load_model(path+'models/model-initial.hdf5')
    # model.load_weights(path+'models/weights-100.hdf5')
    # print model.summary()
    # print model
    # model.save('/home/hpc/capm/sn0515/keras_to_tensorflow/energy.h5')
    # exit()


    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)
    print graph

    # for op in graph.get_operations():
    #     print(op.name)

    x1 = graph.get_tensor_by_name('prefix/Wire_1:0')
    x2 = graph.get_tensor_by_name('prefix/Wire_2:0')
    try:
        y = graph.get_tensor_by_name('prefix/Output/Softmax:0')
        # y = graph.get_tensor_by_name('prefix/Output_Top/Softmax:0')
    except:
        y = graph.get_tensor_by_name('prefix/Output/Relu:0')
        # y = graph.get_tensor_by_name('prefix/Output_Top/Relu:0')
    print x1,x2,y


    # files = ['/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/0-shuffled.hdf5']
    # gen = generate_batches_from_files(files, 100, wires='U', class_type=None, f_size=None, select_dict={}, yield_mc_info=0)
    files = ['/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_Uni_MC_P1/0-shuffled.hdf5']
    # files = ['/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_reduced_MC_P1/0-shuffled.hdf5']
    gen = generate_batches_from_files(files, 100, wires='U', class_type=None, f_size=None, select_dict={}, yield_mc_info=0)
    # files = ['/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_reduced_MC_P1//0-shuffled.hdf5']
    # gen = generate_batches_from_files(files, 100, wires='small', class_type=None, f_size=None, select_dict={}, yield_mc_info=0)

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        wfs = gen.next()
        y_out1 = sess.run(y, feed_dict={x1: wfs[0], x2: wfs[1]})
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        # y_out2 = model.predict_on_batch(wfs)

        # print y_out1#, y_out2
        # for j in range(len(y_out1)):
        #     print y_out1[j, 1], y_out2[j, 1]

    run = '180906-1938/190409-0844' #raw Phase 1
    epoch = '040'  # raw Phase 1
    model_path = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/%s/models/' % (run)
    model = ks.models.load_model(model_path + 'model-000.hdf5')

    # run = '190321-1801/190322-1909'  # energy Phase 1
    # epoch = '137'  # energy Phase 1
    # model_path = '/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/%s/models/' % (run)
    # model = ks.models.load_model(model_path + 'model-initial.hdf5')

    model.load_weights(model_path+'weights-%s.hdf5'%(epoch))
    print model
    y_out2 = model.predict_on_batch(wfs)

    for i in range(len(y_out1)):
        try:
            print y_out1[i,1], '\t', y_out2[i,1], '\t', 100.*(y_out1[i,1]-y_out2[i,1]), '%'
        except:
            print y_out1[i], '\t', y_out2[i], '\t', 100. * (y_out1[i]-y_out2[i]), '%'

    # print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    # sess_config = tf.ConfigProto(gpu_options=gpu_options)
    # persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    # app.run()