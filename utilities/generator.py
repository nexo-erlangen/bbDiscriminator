#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generator used for training a CNN."""

import warnings
import numpy as np
import h5py
import random
import cPickle as pickle
import os
from datetime import datetime

def main():
    files = ['/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/0-shuffled.hdf5',
             '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/1-shuffled.hdf5',
             '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/2-shuffled.hdf5',
             '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/3-shuffled.hdf5',
             '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/4-shuffled.hdf5']
    batchsize = 50
    wires = 'U'
    class_type = 'binary_bb_gamma'
    gen = generate_batches_from_files(files, batchsize, wires, class_type, f_size=None, yield_mc_info=-1)
    # for i in range(100):
    # print sum(gen.next() for i in range(100))
    print getNumEventsFromGen(gen)


#------------- Function used for supplying images to the GPU -------------#
def generate_batches_from_files(files, batchsize, wires=None, class_type=None, f_size=None, select_dict={}, yield_mc_info=0):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string files: Full filepath of the input h5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param int batchsize: Size of the batches that should be generated.
    :param str class_type: String identifier to specify the exact target variables. i.e. 'binary_bb_gamma'
    :param int/None f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param dict select_dict: Dict that contains event selection criteria. e.g. CCIsSS=1.
    :param int yield_mc_info: Specifies if mc-infos should be yielded. 0: Only Waveforms, 1: Waveforms+MC Info, 2: Only MC Info
                               The mc-infos are used for evaluation after training and testing is finished.
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info depending on yield_mc_info).
    """



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
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), "r")
            if f_size is None: f_size = getNumEvents(filename)

            for key in f.keys():
                if key in ['wfs']: continue
                eventInfo[key] = np.asarray(f[key])
            ys = encode_targets(eventInfo, f_size, class_type)

            lst = select_events(eventInfo, select_dict=select_dict, shuffle=True)
            # lst = np.arange(0, f_size, batchsize)
            # random.shuffle(lst)

            if not yield_mc_info in [-1,2]:
                xs = np.asarray(f['wfs'])[:, wireindex]
                # print xs.shape
                # exit()

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

            for i in np.arange(0, lst.size, batchsize):
                batch = sorted(lst[i: i + batchsize])
                # if len(batch) != batchsize: continue

                if not yield_mc_info in [-1,2]:
                    #TODO this line for baseline U-only 2x(Bx350x38x1)
                    xs_i = xs[:, batch]

                    #TODO this line for U-only or UV (Bx350x38/76x1/2/4)
                    # xs_i = xs[batch]

                    ys_i = ys[batch]
                    # w = np.ones(len(batch), dtype='float32')
                    # w_dict = {0: 1.765544106559437, 1: 0.6975434903487171} #test
                    # for id in [0,1]:
                    #     for i in np.nonzero(ys_i[:, id]==1):
                    #         w[i] = w_dict[id]

                    # xs_i_aux = np.dstack((eventInfo['CCPosU'][batch],
                    #                       eventInfo['CCPosV'][batch],
                    #                       eventInfo['CCPosZ'][batch],
                    #                       eventInfo['CCCorrectedEnergy'][batch]))
                    # xs_i_aux = xs_i_aux[..., np.newaxis]

                # if yield_mc_info == 0: yield (xs_i_aux, ys_i)
                # elif yield_mc_info == 1: yield (xs_i_aux, ys_i) + ({key: eventInfo[key][batch] for key in eventInfo.keys()},)
                # if yield_mc_info == 0: yield ([xs_i, xs_i_aux], ys_i)
                # elif yield_mc_info == 1: yield ([xs_i, xs_i_aux], ys_i) + ({key: eventInfo[key][batch] for key in eventInfo.keys()},)
                # if yield_mc_info == 0: yield ([xs_i[0], xs_i[1], xs_i_aux], [ys_i, ys_i])
                # elif yield_mc_info == 1: yield ([xs_i[0], xs_i[1], xs_i_aux], [ys_i, ys_i]) + ({key: eventInfo[key][batch] for key in eventInfo.keys()},)
                # if yield_mc_info == 0: yield (xs_i, ys_i)
                # elif yield_mc_info == 1: yield (xs_i, ys_i) + ({key: eventInfo[key][batch] for key in eventInfo.keys()},)
                if   yield_mc_info == 0:    yield (list(xs_i), ys_i)
                elif yield_mc_info == 1:    yield (list(xs_i), ys_i) + ({ key: eventInfo[key][batch] for key in eventInfo.keys() },)
                elif yield_mc_info == 2:    yield { key: eventInfo[key][batch] for key in eventInfo.keys() }
                elif yield_mc_info == -1:   yield len(batch)
                else:   raise ValueError("Wrong argument for yield_mc_info (-1/0/1/2)")
            f.close()
        if yield_mc_info != 0:
            print 'repeating file list in generator!'
            raise StopIteration

def encode_targets(y_dict, batchsize, class_type=None):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. binary_bb_gamma
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """

    if class_type == None:
        train_y = np.zeros(batchsize, dtype='float32')
    elif class_type == 'binary_bb_gamma':
        from keras.utils import to_categorical
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['ID']  # event ID (0: gamma, 1: bb)
        train_y = to_categorical(train_y, 2)  # convert to one-hot vectors
    elif class_type == 'energy':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['MCEnergy']
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')
    return train_y

def select_events(data_dict, select_dict={}, shuffle=True):
    """
    Encodes the labels (classes) of the images.
    :param dict data_dict: Dictionary that contains ALL event class information for the events.
    :param dict select_dict: Dictionary that contains keys to select events with their values (or low/up limit for range selections).
    :param bool shuffle: Boolean to specify whether the index output list should be shuffled.
    :return: list lst: List that holds the events indices that pass the given selection criteria.
    """

    mask = np.ones(data_dict.values()[0].shape[0], dtype=bool)
    for key, value in select_dict.items():
        if key not in data_dict.keys():
            raise ValueError('Key not in data dict: %s'%(key))
        if isinstance(value, list) and len(value) == 1:
            mask = mask & (data_dict[key] == value[0])
        elif isinstance(value, list) and len(value) == 2:
            mask = mask & (data_dict[key] >= value[0]) & (data_dict[key] < value[1])
        else:
            raise ValueError('Key/Value pair is strange. key: %s . value: %s)'%(key, value))
    lst = np.squeeze(np.argwhere(mask))
    if shuffle:
        random.shuffle(lst)
    return lst

def read_EventInfo_from_files(files, maxNumEvents=0):
    """
    Returns EventInfo dict from a single/list h5py file(s).
    :param string files: Full filepath of the input h5 file, e.g. '[/path/to/file/file.hdf5]'.
    :return: dict eventInfo: Yields a dict which contains the stored mc/data info.
    """

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variable need to be list/np.array/str/dict[dict]')

    if maxNumEvents < 0: raise ValueError('Maximum number of events should be larger 0 (or zero for all)')

    eventInfo = {}
    for idx, filename in enumerate(files):
        f = h5py.File(str(filename), "r")
        for key in f.keys():
            if key in ['wfs', 'gains']: continue
            if idx == 0:
                eventInfo[key] = np.asarray(f[key])
            else:
                eventInfo[key] = np.concatenate((eventInfo[key], np.asarray(f[key])))
        f.close()
        if maxNumEvents > 0 and len(eventInfo.values()[0]) >= maxNumEvents: break
    if maxNumEvents == 0 or len(eventInfo.values()[0]) <= maxNumEvents:
        return eventInfo
    else:
        return { key: value[ 0 : maxNumEvents ] for key,value in eventInfo.items() }

def write_dict_to_hdf5_file(data, file, keys_to_write=['all']):
    """
    Write dict to hdf5 file
    :param dict data: dict containing data.
    :param string file: Full filepath of the output hdf5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param list keys_to_write: Keys that will be written to file
    """
    if not isinstance(data, dict) or not isinstance(file, basestring):
        raise TypeError('passed data/file need to be dict/str. Passed type are: %s/%s'%(type(data),type(file)))
    if 'all' in keys_to_write:
        keys_to_write = data.keys()

    fOUT = h5py.File(file, "w")
    for key in keys_to_write:
        print 'writing', key
        if key not in data.keys():
            print keys_to_write, '\n', data.keys()
            raise ValueError('%s not in dict!'%(str(key)))
        fOUT.create_dataset(key, data=np.asarray(data[key]), dtype=np.float32)
    fOUT.close()
    return

def read_hdf5_file_to_dict(file, keys_to_read=['all']):
    """
    Write dict to hdf5 file
    :param string file: Full filepath of the output hdf5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param list keys_to_write: Keys that will be written to file
    :return dict data: dict containing data.
    """
    data = {}
    fIN = h5py.File(file, "r")
    if 'all' in keys_to_read:
        keys_to_read = fIN.keys()
    for key in keys_to_read:
        if key not in fIN.keys():
            print keys_to_read, '\n', fIN.keys()
            raise ValueError('%s not in file!' % (str(key)))
        data[key] = np.asarray(fIN.get(key))
    fIN.close()
    return data

def predict_events(model, generator):
    X, Y_TRUE, EVENT_INFO = generator.next()
    EVENT_INFO['DNNPred'] = np.asarray(model.predict(X, 50))
    EVENT_INFO['DNNTrue'] = np.asarray(Y_TRUE)
    if len(EVENT_INFO['DNNTrue'].shape) == 3: EVENT_INFO['DNNTrue'] = np.swapaxes(EVENT_INFO['DNNTrue'], 0, 1)
    if len(EVENT_INFO['DNNPred'].shape) == 3: EVENT_INFO['DNNPred'] = np.swapaxes(EVENT_INFO['DNNPred'], 0, 1)
    return EVENT_INFO

def get_events(args, files, model, fOUT):
    # print fOUT
    # fOUT = os.path.splitext(fOUT)[0] + '.p'
    # print fOUT
    try:
        if args.new: raise IOError
        file_ending =  os.path.splitext(fOUT)[1]
        if file_ending == '.p': #just for backwards compatibility
            EVENT_INFO = pickle.load(open(fOUT, "rb"))
            write_dict_to_hdf5_file(data=EVENT_INFO, file=(os.path.splitext(fOUT)[0]+'.hdf5'))
        elif file_ending == '.hdf5':
            EVENT_INFO = read_hdf5_file_to_dict(fOUT)
        else:
            raise ValueError('file ending should be .p/.hdf5 but is %s'%(file_ending))
        if args.events > EVENT_INFO.values()[0].shape[0]: raise IOError
    except IOError:
        events_per_batch = 1000
        if model == None:
            raise SystemError('model not found and not events file found')
        if args.events == 0:
            args.events = getNumEvents(files)
        if args.events % events_per_batch != 0:
            raise ValueError('choose event number in multiples of %f events'%(events_per_batch))

        iterations = round_down(args.events, events_per_batch) / events_per_batch
        gen = generate_batches_from_files(files, events_per_batch, wires=args.wires, class_type=args.var_targets, f_size=None, select_dict=args.select_dict, yield_mc_info=1)

        for i in xrange(iterations):
            print i*events_per_batch, ' of ', iterations*events_per_batch
            EVENT_INFO_temp = predict_events(model, gen)
            if i == 0: EVENT_INFO = EVENT_INFO_temp
            else:
                for key in EVENT_INFO:
                    EVENT_INFO[key] = np.concatenate((EVENT_INFO[key], EVENT_INFO_temp[key]))
        # For now, only class probabilities. For final class predictions, do y_classes = y_prob.argmax(axis=-1)
        EVENT_INFO['DNNPredClass'] = EVENT_INFO['DNNPred'].argmax(axis=-1)
        EVENT_INFO['DNNTrueClass'] = EVENT_INFO['DNNTrue'].argmax(axis=-1)
        EVENT_INFO['DNNPredTrueClass'] = EVENT_INFO['DNNPred'][..., 1]

        # print EVENT_INFO['DNNPredClass'].shape, EVENT_INFO['DNNPred'].shape
        # print EVENT_INFO['DNNTrueClass'].shape, EVENT_INFO['DNNTrue'].shape
        # print EVENT_INFO['DNNPredTrueClass'].shape, EVENT_INFO['DNNPred'].shape
        # exit()

        # for i in range(100):
        #     print i, EVENT_INFO['CCIsSS'][i], EVENT_INFO['DNNTrueClass'][i], EVENT_INFO['DNNPredTrueClass'][i]
        # exit()
        write_dict_to_hdf5_file(data=EVENT_INFO, file=fOUT)
    return EVENT_INFO

def getNumEventsFromGen(gen):
    counter = 0
    while gen:
        try:
            counter += gen.next()
        except StopIteration:
            return counter

def getNumEvents(files):
    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x,y: x+y,files.values())
    else: raise TypeError('passed variable need to be list/np.array/str/dict[dict]')

    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f.values()[0].shape[0]
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

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
    print '===================================== Program finished =============================='