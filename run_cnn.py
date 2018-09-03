#!/usr/bin/env python

import numpy as np
import keras as ks
from keras import backend as K

from utilities.input_utilities import *
from utilities.generator import *
from utilities.cnn_utilities import *
from plot_scripts.plot_input_plots import *
from models.shared_conv import *
from plot_scripts.plot_traininghistory import *

def main(args):
    frac_train = {'mixedUniMC': 0.95}
    frac_val   = {'mixedUniMC': 0.05}
    # frac_train = {'mixedUniMC': 0.50}
    # frac_val = {'mixedUniMC': 0.05}

    splitted_files = splitFiles(args, mode=args.mode, frac_train=frac_train, frac_val=frac_val)

    #TODO InputCorrelationPlot with E/X/Y/Z and signal/bkgd in different colors
    # plotInputCorrelation(args, splitted_files['train'], add='train')
    # plotInputCorrelation(args, splitted_files['val'], add='val')

    executeCNN(args, splitted_files, args.var_targets, args.cnn_arch, args.batchsize, (args.num_weights, args.num_epoch),
               mode=args.mode, n_gpu=(args.num_gpu, 'avolkov'), shuffle=(False, None), tb_logger=args.tb_logger)

    print 'final plots \t start'
    # plot.final_plots(folderOUT=args.folderOUT, obs=pickle.load(open(args.folderOUT + "save.p", "rb")))
    # plot_traininghistory(args)

    print 'final plots \t end'



def executeCNN(args, files, var_targets, nn_arch, batchsize, epoch, mode, n_gpu=(1, 'avolkov'), shuffle=(False, None), tb_logger=False):
    """
    Runs a convolutional neural network.
    :param class args: Contains parsed info about the run.
    :param dict(list) files: Declares the number of bins for each dimension (x,y,z,t) in the train- and testfiles. Can contain multiple n_bins tuples.
                               Multiple n_bins tuples are currently only used for multi-input models with multiple input files per batch.
    :param str var_targets: Declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param str nn_arch: Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    :param int batchsize: Batchsize that should be used for the cnn.
    :param (int/int) epoch: Declares if a previously trained model or a new model (=0) should be loaded.
    :param str mode: Specifies what the function should do - train & test a model or evaluate (on mc/data) a 'finished' model?
                     Currently, there are three modes available: 'train' & 'mc' & 'data'.
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param (bool, None/int) shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param bool tb_logger: Declares if a tb_callback should be used during training (takes longer to train due to overhead!).
    """

    print '\nEpoch Interval:\t', epoch[0], ' - ', epoch[1], '\n'

    if epoch[0] == 0:
        if nn_arch == 'DCNN':
            model = create_shared_dcnn_network_U()
        elif nn_arch == 'ResNet':
            raise ValueError('Currently, this is not implemented')
            # model = create_vgg_like_model(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
            #                               n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128),
            #                               swap_4d_channels=swap_4d_channels)
        elif nn_arch == 'Inception':
            model = create_shared_inceptionV3_network_U()
        elif nn_arch == 'Conv_LSTM':
            raise ValueError('Currently, this is not implemented')
            # model = create_convolutional_lstm(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
            #                                   n_filters=(16, 16, 32, 32, 32, 32, 64, 64))
        else:
            raise ValueError('Currently, only "DCNN" are available as nn_arch')
    else:
        # raise ValueError('Check, if loading models is implemented yet')
        model = load_trained_model(args)
        # model = ks.models.load_model(
            # 'models/trained/trained_' + modelname + '_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5')

    if mode == 'train':
        model.summary()

        # plot model, install missing packages with conda install if it throws a module error
        try:
            ks.utils.plot_model(model, to_file=args.folderOUT + '/plot_model.png', show_shapes=True,
                                show_layer_names=True)
        except OSError:
            save_plot_model_script(folderOUT=args.folderOUT)
            print 'could not produce plot_model.png ---- run generate_model_plot on CPU'

        # exit()

        adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  # epsilon=1 for deep networks
        optimizer = adam  # Choose optimizer, only used if epoch == 0

        # model, batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize)  # TODO compile after restart????
        # if n_gpu[0] > 1: model.compile(loss=loss_opt[0], optimizer=optimizer, metrics=[loss_opt[1]])  # TODO check

        # lr_metric = get_lr_metric(optimizer)

        if epoch[0] == 0:
            print 'Compiling Keras model\n'
            # model.compile(
            #     loss='mean_squared_error',
            #     optimizer=optimizer,
            #     metrics=['mean_absolute_error'])  # , lr_metric])
            model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])  # , lr_metric])

        print "\nTraining begins in Epoch:\t", epoch

        model.save(args.folderOUT + "models/model-000.hdf5")
        model.save_weights(args.folderOUT + "models/weights-000.hdf5")
        model = fit_model(args, model, files, batchsize, var_targets, epoch, shuffle, n_events=None, tb_logger=tb_logger)
        model.save_weights(args.folderOUT + "models/weights_final.hdf5")
        model.save(args.folderOUT + "models/model_final.hdf5")
    elif mode == 'mc':
        print 'Validate on MC events'

        args.sources = "".join(sorted(args.sources))
        args.position = "".join(sorted(args.position))

        args.folderOUT += "1validation-data/" + args.sources + "-" + args.position + "/"
        os.system("mkdir -p -m 770 %s " % (args.folderOUT))
        data = get_events(args=args, files=files, model=model, fOUT=(args.folderOUT + "events_" + str(args.num_weights) + "_" + args.sources + "-" + args.position + ".p"))

        # validation_mc_plots(folderOUT=args.folderOUT, data=data, epoch=args.num_weights, sources=args.sources, position=args.position)
    elif mode == 'data':
        print 'Validate on real data events'

        args.sources = "".join(sorted(args.sources))
        args.position = "".join(sorted(args.position))

        args.folderOUT += "0physics-data/" + str(args.num_weights) + '-' + args.sources + "-" + args.position + "/"
        os.system("mkdir -p -m 770 %s " % (args.folderOUT))
        data = get_events(args=args, files=files, model=model, fOUT=(
        args.folderOUT + "events_" + str(args.num_weights) + "_" + args.sources + "-" + args.position + ".p"))

        validation_data_plots(folderOUT=args.folderOUT, data=data, epoch=args.num_weights, sources=args.sources,
                              position=args.position)

        # raise ValueError('Check, if model mc evaluation is implemented yet')
        # After training is finished, investigate model performance
        # arr_energy_correct = make_performance_array_energy_correct(model, test_files[0][0], n_bins, class_type,
        #                                                            batchsize, xs_mean, swap_4d_channels, str_ident,
        #                                                            samples=None)
        # np.save('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy', arr_energy_correct)
        #
        # arr_energy_correct = np.load('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy')
        # # arr_energy_correct = np.load('results/plots/saved_predictions/backup/arr_energy_correct_model_VGG_4d_xyz-t-tight-1-w-geo-fix_and_yzt-x-tight-1-wout-geo-fix_and_4d_xyzt_muon-CC_to_elec-CC_double_input_single_train.npy')
        # make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct,
        #                                               title='Classification for muon-CC_and_elec-CC_3-100GeV',
        #                                               filename='results/plots/PT_' + modelname,
        #                                               compare_pheid=True)  # TODO think about more automatic savenames
        # make_prob_hists(arr_energy_correct[:, ], modelname=modelname, compare_pheid=True)
        # make_property_to_accuracy_plot(arr_energy_correct, 'bjorken-y',
        #                                title='Bjorken-y distribution vs Accuracy, 3-100GeV',
        #                                filename='results/plots/PT_bjorken_y_vs_accuracy' + modelname, e_cut=False,
        #                                compare_pheid=True)
        #
        # make_hist_2d_property_vs_property(arr_energy_correct, modelname,
        #                                   property_types=('bjorken-y', 'probability'), e_cut=(3, 100),
        #                                   compare_pheid=True)
        # calculate_and_plot_correlation(arr_energy_correct, modelname, compare_pheid=True)
    else:
        raise ValueError('chosen mode not available. Choose between train/eval')


# ----------------------------------------------------------
# Define model
# ----------------------------------------------------------
def schedule_learning_rate(model, epoch, lr_initial=0.001, n_gpu=1, manual_mode=(False, None, 0.0, None)):
    """
    Function that schedules a learning rate during training.
    If manual_mode[0] is False, the current lr will be automatically calculated if the training is resumed, based on the epoch variable.
    If manual_mode[0] is True, the final lr during the last training session (manual_mode[1]) and the lr_decay (manual_mode[1])
    have to be set manually.
    :param Model model: Keras nn model instance. Used for setting the lr.
    :param int epoch: The epoch number at which this training session is resumed (last finished epoch).
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param list train_files: list of tuples that contains the trainfiles and their number of rows (filepath, f_size).
    :param float lr_initial: Initial lr that is used with the automatic mode. Typically 0.01 for SGD and 0.001 for Adam.
    :param (bool, None/float, float, None/float) manual_mode: Tuple that controls the options for the manual mode.
            manual_mode[0] = flag to enable the manual mode, manual_mode[1] = lr value, of which the mode should start off
            manual_mode[2] = lr_decay during epochs, manual_mode[3] = current lr, only used to check if this is the first instance of the while loop
    :return: int epoch: The epoch number of the new epoch (+= 1).
    :return: float lr: Learning rate that has been set for the model and for this epoch.
    :return: float lr_decay: Learning rate decay that has been used to decay the lr rate used for this epoch.
    """

    if manual_mode[0] is True:
        raise ValueError('Currently, no manuel_mode == True implemented')
        # lr = manual_mode[1] if manual_mode[3] is None else K.get_value(model.optimizer.lr)
        # lr_decay = manual_mode[2]
        # K.set_value(model.optimizer.lr, lr)
        #
        # if epoch[0] > 1 and lr_decay > 0:
        #     lr *= 1 - float(lr_decay)
        #     K.set_value(model.optimizer.lr, lr)
        #     print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
        #           ' before epoch ' + str(epoch[0]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')'
    else:
        if epoch == 0:
            lr, lr_decay = lr_initial, 0.00
            #lr, lr_decay = lr_initial * n_gpu[0], 0.00
            K.set_value(model.optimizer.lr, lr)
            print 'Set learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch)
        else:
            lr, lr_decay = get_new_learning_rate(epoch, lr_initial, n_gpu)
            K.set_value(model.optimizer.lr, lr)
            print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
                  ' before epoch ' + str(epoch) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')'

    return epoch, lr, lr_decay

def get_new_learning_rate(epoch, lr_initial, n_gpu):
    """
    Function that calculates the current learning rate based on the number of already trained epochs.
    Learning rate schedule is as follows: lr_decay = 7% for lr > 0.0003
                                          lr_decay = 4% for 0.0003 >= lr > 0.0001
                                          lr_decay = 2% for 0.0001 >= lr
    :param int epoch: The number of the current epoch which is used to calculate the new learning rate.
    :param float lr_initial: Initial lr for the first epoch. Typically 0.01 for SGD and 0.001 for Adam.
    :param int n_gpu: number of gpu's that are used during the training. Used for scaling the lr.
    :return: float lr_temp: Calculated learning rate for this epoch.
    :return: float lr_decay: Latest learning rate decay used.
    """
    n_lr_decays = epoch - 1
    lr_temp = lr_initial # * n_gpu TODO think about multi gpu lr
    lr_decay = None

    for i in xrange(n_lr_decays):

        if lr_temp > 0.0003:
            lr_decay = 0.07
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04
        else:
            lr_decay = 0.02

        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp, lr_decay

def fit_model(args, model, files, batchsize, var_targets, epoch, shuffle, n_events=None, tb_logger=False):
    """
    Trains a model based on the Keras fit_generator method.
    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.
    :param class args: Contains parsed info about the run.
    :param ks.model.Model model: Keras model of a neural network.
    :param dict(list) files: dict containing filepaths of the files that should be used for training/validation/testing.
    :param int batchsize: Batchsize that is used in the fit_generator method.
    :param str var_targets: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param int epoch: Epoch of the model if it has been trained before.
    :param (bool, None/int) shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for training.
    :param bool tb_logger: Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log!).
    """

    train_steps_per_epoch = int(getNumEvents(files['train']) / batchsize)
    validation_steps = int(min([getNumEvents(files['val']), 5000]) / batchsize)
    genVal = generate_batches_from_files(files['val'], batchsize=batchsize, class_type=var_targets, yield_mc_info=0)

    callbacks = []
    csvlogger = ks.callbacks.CSVLogger(args.folderOUT + 'history.csv', separator='\t', append=args.resume)
    modellogger = ks.callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:03d}.hdf5', save_weights_only=True, period=1)
    lrscheduler = ks.callbacks.LearningRateScheduler(LRschedule_stepdecay, verbose=1)
    epochlogger = EpochLevelPerformanceLogger(args=args, files=files['val'], var_targets=var_targets)
    batchlogger = BatchLevelPerformanceLogger(display=25, skipBatchesVal=20, steps_per_epoch=train_steps_per_epoch, args=args,
                                              genVal=generate_batches_from_files(files['val'], batchsize=batchsize//2, class_type=var_targets, yield_mc_info=0))
    callbacks.append(csvlogger)
    callbacks.append(modellogger)
    callbacks.append(lrscheduler)
    callbacks.append(batchlogger)
    # callbacks.append(epochlogger)
    if tb_logger is True:
        print 'TensorBoard Log Directory:'
        print args.folderRUNS + 'tb_logs/%s'%(args.folderOUT[args.folderOUT.rindex('/', 0, len(args.folderOUT) - 1) + 1 : -1])
        tensorlogger = TensorBoardWrapper(generate_batches_from_files(files['val'], batchsize=batchsize, class_type=var_targets, yield_mc_info=0),
                                          nb_steps=validation_steps, log_dir=(args.folderRUNS + 'tb_logs/%s'%(args.folderOUT[args.folderOUT.rindex('/', 0, len(args.folderOUT) - 1) + 1 : -1])),
                                          histogram_freq=1, batch_size=batchsize, write_graph=True, write_grads=True, write_images=True)
        callbacks.append(tensorlogger)

    print 'Set learning rate to ' + str(K.get_value(model.optimizer.lr))

    epoch = (int(epoch[0]), int(epoch[1]))
    print 'training from:', epoch

    print 'training steps:', train_steps_per_epoch
    print 'validation steps:', validation_steps

    model.fit_generator(
        generate_batches_from_files(files['train'], batchsize=batchsize, class_type=var_targets, yield_mc_info=0),
        steps_per_epoch=train_steps_per_epoch,
        epochs=epoch[0]+epoch[1],
        initial_epoch=epoch[0],
        verbose=1,
        max_queue_size=10,
        validation_data=genVal,
        validation_steps=validation_steps,
        callbacks=callbacks)

    print 'Model performance\tloss\t\taccuracy'
    print '\tTrain:\t\t%.4f\t\t%.4f'    % tuple(model.evaluate_generator(generate_batches_from_files(files['train'], batchsize, var_targets), steps=50))
    print '\tValid:\t\t%.4f\t\t%.4f'    % tuple(model.evaluate_generator(generate_batches_from_files(files['val']  , batchsize, var_targets), steps=50))
    return model

def load_trained_model(args):
    nb_weights = args.num_weights
    print "===================================== load Model ==============================================="
    try:
        print "%smodels/(model/weights)-%s.hdf5" % (args.folderMODEL, nb_weights)
        print "================================================================================================\n"
        try:
            model = ks.models.load_model(args.folderMODEL + "models/model-000.hdf5")
            model.load_weights(args.folderMODEL + "models/weights-" + nb_weights + ".hdf5")
        except:
            model = ks.models.load_model(args.folderMODEL + "models/model-" + nb_weights + ".hdf5")
        os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT + "history.csv"))
        if nb_weights=='final':
            epoch_start = 1+int(np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)['epoch'][-1])
            print epoch_start
        else:
            epoch_start = 1+int(nb_weights)
    except Exception:
        raise Exception("\t\tMODEL NOT FOUND!\n")
    return model

def LRschedule_stepdecay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lrate

def save_plot_model_script(folderOUT):
    """
    Function for saving python script for producing model_plot.png
    """
    with open(folderOUT+'generate_model_plot.py', 'w') as f_out:
        f_out.write('#!/usr/bin/env python' + '\n')
        f_out.write('try:' + '\n')
        f_out.write('\timport keras as ks' + '\n')
        f_out.write('except ImportError:' + '\n')
        f_out.write('\tprint "Keras not available. Activate tensorflow_cpu environment"' + '\n')
        f_out.write('\traise SystemExit("=========== Error -- Exiting the script ===========")' + '\n')
        f_out.write('model = ks.models.load_model("%smodels/model-000.hdf5")'%(folderOUT) + '\n')
        f_out.write('try:' + '\n')
        f_out.write('\tks.utils.plot_model(model, to_file="%s/plot_model.png", show_shapes=True, show_layer_names=True)'%(folderOUT) + '\n')
        f_out.write('except OSError:' + '\n')
        f_out.write('\tprint "could not produce plot_model.png ---- try on CPU"' + '\n')
        f_out.write('\traise SystemExit("=========== Error -- Exiting the script ===========")' + '\n')
        f_out.write('print "=========== Generating Plot Finished ==========="' + '\n')
        f_out.write('\n')

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':

    args = parseInput()

    try:
        main(args=args)
    except KeyboardInterrupt:
        print ' >> Interrupted << '
    finally:
        adjustPermissions(args.folderOUT)

    print '===================================== Program finished =============================='
