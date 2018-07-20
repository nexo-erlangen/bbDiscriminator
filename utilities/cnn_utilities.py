#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import keras as ks
import os
from sys import path
from keras import backend as K

path.append('/home/hpc/capm/sn0515/UVWireRecon')

from utilities.input_utilities import *
from utilities.generator import *
from utilities.cnn_utilities import *
from plot_scripts.plot_input_plots import *
from plot_scripts.plot_traininghistory import *
from plot_scripts.plot_validation import *

class TensorBoardWrapper(ks.callbacks.TensorBoard):
    """Up to now (05.10.17), Keras doesn't accept TensorBoard callbacks with validation data that is fed by a generator.
     Supplying the validation data is needed for the histogram_freq > 1 argument in the TB callback.
     Without a workaround, only scalar values (e.g. loss, accuracy) and the computational graph of the model can be saved.

     This class acts as a Wrapper for the ks.callbacks.TensorBoard class in such a way,
     that the whole validation data is put into a single array by using the generator.
     Then, the single array is used in the validation steps. This workaround is experimental!"""
    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in xrange(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class BatchLevelPerformanceLogger(ks.callbacks.Callback):
    # Gibt loss aus über alle :display batches, gemittelt über die letzten :display batches
    def __init__(self, display, steps_per_epoch, args):
        ks.callbacks.Callback.__init__(self)
        self.seen = 0
        self.display = display
        self.averageLoss = 0
        self.averageMAE = 0
        self.averageValLoss = 0
        self.averageValMAE = 0
        self.args = args
        self.logfile_train_fname = self.args.folderOUT + 'log_train.txt'
        self.logfile_train = None
        self.steps_per_epoch = steps_per_epoch
        # self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.averageLoss += logs.get('loss')
        self.averageMAE += logs.get('mean_absolute_error')

        # if self.seen % 100 == 0:
        #     print logs.keys()

        # self.averageValLoss += logs.get('val_loss')
        # self.averageValMAE += logs.get('val_mean_absolute_error')
        if self.seen % self.display == 0:
            averaged_loss = self.averageLoss / self.display
            averaged_mae = self.averageMAE / self.display
            averaged_ValLoss = self.averageValLoss / self.display
            averaged_ValMAE = self.averageValMAE / self.display

            batchnumber_float = (self.seen - self.display / 2.) / float(self.steps_per_epoch) # + self.epoch - 1  # start from zero
            self.loglist.append('\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(self.seen, batchnumber_float, averaged_loss, averaged_mae, averaged_ValLoss, averaged_ValMAE))
            self.averageLoss = 0
            self.averageMAE = 0
            self.averageValLoss = 0
            self.averageValMAE = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.loglist = []

    def on_epoch_end(self, epoch, logs={}):
        self.logfile_train = open(self.logfile_train_fname, 'a+')
        if os.stat(self.logfile_train_fname).st_size == 0: self.logfile_train.write("#Batch\t#Batch_float\tLoss\tMAE\tVal-Loss\tVal-MAE")

        for batch_statistics in self.loglist: # only write finished epochs to the .txt
            self.logfile_train.write(batch_statistics)

        self.logfile_train.flush()
        os.fsync(self.logfile_train.fileno())
        self.logfile_train.close()


class EpochLevelPerformanceLogger(ks.callbacks.Callback):
    def __init__(self, args, files, var_targets):
        ks.callbacks.Callback.__init__(self)
        self.validation_data = None
        self.args = args
        self.files = files
        self.events_val = min([getNumEvents(files), 2000])
        self.events_per_batch = 50
        self.val_iterations = round_down(self.events_val, self.events_per_batch) / self.events_per_batch
        self.val_gen = generate_batches_from_files(files, batchsize=self.events_per_batch, class_type=var_targets, yield_mc_info=1)

    def on_train_begin(self, logs={}):
        self.losses = []
        # self.val_losses = []
        if self.args.resume:
            os.system("cp %s %s" % (self.args.folderMODEL + "save.p", self.args.folderOUT + "save.p"))
        else:
            pickle.dump({}, open(self.args.folderOUT + "save.p", "wb"))
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        Y_PRED, Y_TRUE, EVENT_INFO = [], [], []
        for i in xrange(self.val_iterations):
            Y_PRED_temp, Y_TRUE_temp, EVENT_INFO_temp = predict_events(self.model, self.val_gen)
            Y_PRED.extend(Y_PRED_temp)
            Y_TRUE.extend(Y_TRUE_temp)
            EVENT_INFO.extend(EVENT_INFO_temp)
        # Eval_dict = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': np.asarray(EVENT_INFO)}
        # obs = plot.make_plots(self.args.folderOUT, dataIn=dataIn, epoch=str(epoch), sources='th', position='S5')
        self.dict_out = pickle.load(open(self.args.folderOUT + "save.p", "rb"))
        self.dict_out[epoch] = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': np.asarray(EVENT_INFO),
                                     'loss': logs['loss'], 'mean_absolute_error': logs['mean_absolute_error'],
                                     'val_loss': logs['val_loss'], 'val_mean_absolute_error': logs['val_mean_absolute_error']}
        pickle.dump(self.dict_out, open(self.args.folderOUT + "save.p", "wb"))
        on_epoch_end_plots(folderOUT=self.args.folderOUT, epoch=epoch, data=self.dict_out[epoch])
        # print logs.keys()


        # plot_train_and_test_statistics(modelname)
        # plot_weights_and_activations(test_files[0][0], n_bins, class_type, xs_mean, swap_4d_channels, modelname,
        #                              epoch[0], file_no, str_ident)
        # plot_traininghistory(args)

        return