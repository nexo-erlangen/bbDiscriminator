#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generator used for training a CNN."""

import warnings
import numpy as np
import h5py
import random
import cPickle as pickle

#------------- Function used for supplying images to the GPU -------------#
def generate_batches_from_files(files, batchsize, class_type=None, f_size=None, yield_mc_info=0):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string files: Full filepath of the input h5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param int batchsize: Size of the batches that should be generated.
    :param str class_type: String identifier to specify the exact target variables. i.e. 'energy_and_position'
    :param int/None f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param int yield_mc_info: Specifies if mc-infos should be yielded. 0: Only Waveforms, 1: Waveforms+MC Info, 2: Only MC Info
                               The mc-infos are used for evaluation after training and testing is finished.
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info depending on yield_mc_info).
    """

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    eventInfo = {}
    while 1:
        random.shuffle(files)  # TODO maybe omit in future? # TODO shuffle events between files
        for filename in files:
            f = h5py.File(str(filename), "r")
            if f_size is None: f_size = getNumEvents(filename)
                # warnings.warn( 'f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                #     'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

            lst = np.arange(0, f_size, batchsize)
            # random.shuffle(lst)  #  TODO maybe omit in future?

            # filter the labels we don't want for now
            for key in f.keys():
                if key in ['wfs']: continue
                eventInfo[key] = np.asarray(f[key])
            ys = encode_targets(eventInfo, f_size, class_type)

            for i in lst:
                if not yield_mc_info == 2:
                    xs_i = f['wfs'][ i : i + batchsize ]
                    xs_i = np.swapaxes(xs_i, 0, 1)
                    xs_i = np.swapaxes(xs_i, 2, 3)
                ys_i = ys[i: i + batchsize]
                # print ys_i
                # raw_input('')

                if   yield_mc_info == 0:    yield (list(xs_i), ys_i)
                elif yield_mc_info == 1:    yield (list(xs_i), ys_i) + ({ key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() },)
                elif yield_mc_info == 2:    yield { key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() }
                else:   raise ValueError("Wrong argument for yield_mc_info (0/1/2)")
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize

def encode_targets(y_dict, batchsize, class_type=None):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. energy_and_position
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """

    if class_type == None:
        train_y = np.zeros(batchsize, dtype='float32')
    elif class_type == 'energy_and_position' or class_type == 'position_and_energy':
        train_y = np.zeros((batchsize, 4), dtype='float32')
        train_y[:,0] = y_dict['MCEnergy'][:,0]  # energy
        train_y[:,1] = y_dict['MCPosX'][:,0]  # dir_x
        train_y[:,2] = y_dict['MCPosY'][:,0]  # dir_y
        train_y[:,3] = y_dict['MCTime'][:,0]  # time (to calculate dir_z)
    elif class_type == 'energy':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:,0] = y_dict['MCEnergy'][:,0]  # energy
    elif class_type == 'position_x_y':
        train_y = np.zeros((batchsize, 2), dtype='float32')
        train_y[:,0] = y_dict['MCPosX'][:,0]  # dir_x
        train_y[:,1] = y_dict['MCPosY'][:,0]  # dir_y
    elif class_type == 'time':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['MCTime'][:, 0]  # time (to calculate dir_z)
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')
    return train_y

def predict_events(model, generator):
    X, Y_TRUE, EVENT_INFO = generator.next()
    Y_PRED = np.asarray(model.predict(X, 10))
    return (Y_PRED, Y_TRUE, EVENT_INFO)

def get_events(args, files, model, fOUT):
    try:
        if args.new: raise IOError
        spec = pickle.load(open(fOUT, "rb"))
        if args.events > len(spec['Y_TRUE']): raise IOError
    except IOError:
        if model == None: print 'model not found and not events file found' ; exit()
        events_per_batch = 50
        if args.events % events_per_batch != 0: raise ValueError('choose event number in multiples of %f events'%(events_per_batch))
        iterations = round_down(args.events, events_per_batch) / events_per_batch
        gen = generate_batches_from_files(files, events_per_batch, class_type=args.var_targets, f_size=None, yield_mc_info=1)

        Y_PRED, Y_TRUE = [], []
        for i in xrange(iterations):
            print i*events_per_batch, ' of ', iterations*events_per_batch
            Y_PRED_temp, Y_TRUE_temp, EVENT_INFO_temp = predict_events(model, gen)
            Y_PRED.extend(Y_PRED_temp)
            Y_TRUE.extend(Y_TRUE_temp)
            if i == 0: EVENT_INFO = EVENT_INFO_temp
            else:
                for key in EVENT_INFO:
                    EVENT_INFO[key] = np.concatenate((EVENT_INFO[key], EVENT_INFO_temp[key]))

        spec = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': EVENT_INFO}
        pickle.dump(spec, open(fOUT, "wb"))
    return spec

def getNumEvents(files):
    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x,y: x+y,files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f['MCEventNumber'].shape[0]
        f.close()
    return counter

def get_array_memsize(array, unit='KB'):
    """
    Calculates the approximate memory size of an array.
    :param ndarray array: an array.
    :param string unit: output unit of memsize.
    :return: float memsize: size of the array given in unit.
    """
    units = {'B': 0., 'KB': 1., 'MB': 2., 'GB':3.}
    if isinstance(array, list):
        array = np.asarray(array)

    shape = array.shape
    n_numbers = reduce(lambda x, y: x*y, shape) # number of entries in an array
    precision = array.dtype.itemsize # Precision of each entry in bytes
    memsize = (n_numbers * precision) # in bytes
    return memsize/1024**units[unit]

def round_down(num, divisor):
    return num - (num%divisor)

