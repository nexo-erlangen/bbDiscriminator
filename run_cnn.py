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
from plot_scripts.plot_validation import *

def main(args):
    if len(args.endings.keys()) == 1 and args.mode == 'train':
        frac_train = {args.endings.keys()[0]: 0.90}
        frac_val   = {args.endings.keys()[0]: 0.10}
    else:
        frac_train = {'': 0.0}
        frac_val   = {'': 0.0}

    splitted_files = splitFiles(args, mode=args.mode, frac_train=frac_train, frac_val=frac_val)

    executeCNN(args, splitted_files, args.var_targets, args.cnn_arch, args.batchsize, (args.num_weights, args.num_epoch),
               mode=args.mode, n_gpu=(args.num_gpu, 'avolkov'), shuffle=(False, None), tb_logger=args.tb_logger)


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

    gen_kwargs = {
        'batchsize': batchsize,
        'wires': args.wires,
        'class_type': var_targets,
        'select_dict': args.select_dict
    }

    print '\nEpoch Interval:\t', epoch[0], ' - ', epoch[1], '\n'

    if epoch[0] == 0:
        if nn_arch == 'DCNN':
            if args.wires in ['U', 'V', 'small']:
                model = create_shared_dcnn_network_2()
            elif args.wires in ['UV', 'U+V']:
                model = create_shared_dcnn_network_4()
            else: raise ValueError('passed wire specifier need to be U/V/UV')
        elif nn_arch == 'ResNet':
            raise ValueError('Currently, this is not implemented')
        elif nn_arch == 'Inception':
            if args.wires in ['U', 'V', 'small']:
                # model = create_shared_inception_network_2_extra_input()
                model = create_shared_inception_network_2() # TODO model = create_shared_inception_network_2()
                # model = create_shared_inceptionV4_network_2()
            elif args.wires in ['UV', 'U+V']:
                model = create_shared_inception_network_2() # TODO model = create_shared_inception_network_4()
            else: raise ValueError('passed wire specifier need to be U/V/U+V/UV/small')
        else:
            raise ValueError('Currently, only DCNN and Inception are available as nn_arch')
    else:
        model = load_trained_model(args)

    if mode == 'train':
        # print 'start model'

        # model = create_shared_inception_network_2_extra_input(kwargs_inc={'dropout': 0.1, 'activity_regularizer': regularizers.l2(0.01)})

        # model = ks.models.load_model('/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/test/models/weights-000.hdf5')
        # model.load_weights('/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/test/models/weights-023.hdf5', by_name=False)

        # model = create_shared_inception_network_2_extra_input()
        # # model = create_shared_inception_network_2_extra_input(kwargs_inc={'dropout': 0.1})
        # model.load_weights(args.folderMODEL + "models/weights-" + str(args.num_weights).zfill(3) + ".hdf5", by_name=True)
        #
        # for layer in model.layers:
        #     # print layer.name, layer.trainable
        #     if not layer.name in ['31', '32', '33', '21', '22', 'Flatten_Pos', 'Output', 'Aux_Input']:
        #         layer.trainable = False
        #     if layer.trainable == True:
        #         print layer.name, layer.trainable
        #
        # print 'end model'

        model.summary()
        try: # plot model, install missing packages with conda install if it throws a module error
            ks.utils.plot_model(model, to_file=args.folderOUT + '/plot_model.png',
                                show_shapes=True, show_layer_names=False)
        except OSError:
            save_plot_model_script(folderOUT=args.folderOUT)
            print '\n\ncould not produce plot_model.png ---- run generate_model_plot on CPU'

        adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)  # Default: epsi: None, Deep NN: epsi=0.1/1.0
        optimizer = adam  # Choose optimizer, only used if epoch == 0

        # model, batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize)  # TODO compile after restart????
        # if n_gpu[0] > 1: model.compile(loss=loss_opt[0], optimizer=optimizer, metrics=[loss_opt[1]])  # TODO check

        if epoch[0] == 0: # or True: #TODO or True only for manual adding layers to freezed network
            print 'Compiling Keras model\n'
            model.compile(
                loss='categorical_crossentropy',
                # loss_weights=[1., 0.0],
                optimizer=optimizer,
                metrics=['accuracy'])
            # TODO Add Precision/Recall to metric, see:
            # TODO https://stackoverflow.com/questions/43076609/how-to-calculate-precision-and-recall-in-keras

        print 'optimizer epsilon:', model.optimizer.epsilon

        print "\nTraining begins in Epoch:\t", epoch

        model.save(args.folderOUT + "models/model-000.hdf5")
        model.save_weights(args.folderOUT + "models/weights-000.hdf5")
        model = fit_model(args, model, files, batchsize, var_targets, epoch, shuffle, n_events=None, tb_logger=tb_logger)
        model.save_weights(args.folderOUT + "models/weights_final.hdf5")
        model.save(args.folderOUT + "models/model_final.hdf5")
    elif mode in ['mc', 'data']:
        print 'Validate on %s events'%(mode)

        args.sources = "".join(sorted(args.sources))
        args.position = "".join(sorted(args.position))
        args.folderOUT += "0validation/" + args.sources + "-" + mode + "-" + args.position + "-" + str(args.num_weights) + "-" + args.wires + "/"
        os.system("mkdir -p -m 770 %s " % (args.folderOUT))

        EVENT_INFO = get_events(args=args, files=files, model=model,
                          fOUT=(args.folderOUT + "events_" + str(args.num_weights).zfill(3) + "_" + args.sources + "-" + mode + "-" + args.position + "-" + args.wires + ".hdf5"))

        validation_mc_plots(args=args, folderOUT=args.folderOUT, data=EVENT_INFO)
    else:
        raise ValueError('chosen mode (%s) not available. Choose between train/mc/data'%(mode))


# ----------------------------------------------------------
# Train model
# ----------------------------------------------------------
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
    gen_kwargs = {
        'batchsize': batchsize,
        'wires': args.wires,
        'class_type': var_targets,
        'select_dict': args.select_dict
    }

    train_steps_per_epoch = getNumEventsFromGen(generate_batches_from_files(files['train'], yield_mc_info=-1, **gen_kwargs))//batchsize
    validation_steps = getNumEventsFromGen(generate_batches_from_files(files['val'], yield_mc_info=-1, **gen_kwargs))//batchsize
    genVal = generate_batches_from_files(files['val'],yield_mc_info=0, **gen_kwargs)

    callbacks = []
    csvlogger = ks.callbacks.CSVLogger(args.folderOUT + 'history.csv', separator='\t', append=args.resume)
    modellogger = ks.callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:03d}.hdf5', save_weights_only=True, period=1)
    lrscheduler = ks.callbacks.LearningRateScheduler(LRschedule_stepdecay, verbose=1)
    epochlogger = EpochLevelPerformanceLogger(args=args, files=files['val'], var_targets=var_targets)
    batchlogger = BatchLevelPerformanceLogger(display=10, skipBatchesVal=10, steps_per_epoch=train_steps_per_epoch, args=args, #15, 20
                                              genVal=generate_batches_from_files(files['val'], yield_mc_info=0, **gen_kwargs)) #batchsize//2
    callbacks.append(csvlogger)
    callbacks.append(modellogger)
    callbacks.append(lrscheduler)
    callbacks.append(batchlogger)
    # callbacks.append(epochlogger)
    if tb_logger is True:
        print 'TensorBoard Log Directory:'
        print args.folderRUNS + 'tb_logs/%s'%(args.folderOUT[args.folderOUT.rindex('/', 0, len(args.folderOUT) - 1) + 1 : -1])
        tensorlogger = TensorBoardWrapper(generate_batches_from_files(files['val'], yield_mc_info=0, **gen_kwargs),
                                          nb_steps=validation_steps, log_dir=(args.folderRUNS + 'tb_logs/%s'%(args.folderOUT[args.folderOUT.rindex('/', 0, len(args.folderOUT) - 1) + 1 : -1])),
                                          histogram_freq=1, batch_size=batchsize, write_graph=True, write_grads=True, write_images=True)
        callbacks.append(tensorlogger)

    epoch = (int(epoch[0]), int(epoch[1]))
    print 'training from:', epoch

    print 'training steps:', train_steps_per_epoch
    print 'validation steps:', validation_steps

    model.fit_generator(
        generate_batches_from_files(files['train'], yield_mc_info=0, **gen_kwargs),
        steps_per_epoch=train_steps_per_epoch,
        epochs=epoch[0]+epoch[1],
        initial_epoch=epoch[0],
        verbose=1,
        max_queue_size=10,
        validation_data=genVal,
        validation_steps=validation_steps,
        callbacks=callbacks)

    print 'Model performance\tloss\t\taccuracy'
    print '\tTrain:\t\t%.4f\t\t%.4f'    % tuple(model.evaluate_generator(generate_batches_from_files(files['train'], batchsize, args.wires, var_targets), steps=50))
    print '\tValid:\t\t%.4f\t\t%.4f'    % tuple(model.evaluate_generator(generate_batches_from_files(files['val']  , batchsize, args.wires, var_targets), steps=50))
    return model

def load_trained_model(args):
    nb_weights = str(args.num_weights).zfill(3) #args.num_weights
    print "===================================== load Model ==============================================="
    try:
        print "%smodels/(model/weights)-%s.hdf5" % (args.folderMODEL, nb_weights)
        print "================================================================================================\n"
        try:
            model = ks.models.load_model(args.folderMODEL + "models/model-000.hdf5")
            model.load_weights(args.folderMODEL + "models/weights-" + nb_weights + ".hdf5", by_name=False)
        except:
            model = ks.models.load_model(args.folderMODEL + "models/model-" + nb_weights + ".hdf5")
        if args.folderMODEL != args.folderOUT:
            os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT + "history.csv"))
        if nb_weights=='final':
            epoch_start = 1+int(np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)['epoch'][-1])
            print epoch_start
        else:
            epoch_start = 1+int(nb_weights)
    except Exception:
        raise Exception("\t\tMODEL NOT FOUND!\n")
    return model

def calculate_class_weights(files, gen_kwargs):
    class_weights = {}
    n_samples = 0
    for id in [0,1]:
        gen_kwargs['select_dict']['ID'] = [id]
        num_id = getNumEventsFromGen(generate_batches_from_files(files, yield_mc_info=-1, **gen_kwargs))
        n_samples += num_id
        class_weights[id] = 1./float(num_id)
    class_weights.update({key: n_samples*class_weights[key]/float(len(class_weights.keys())) for key in class_weights.keys()})
    return class_weights

def LRschedule_stepdecay(epoch):
    # initial_lrate = 0.01
    # step_drop = 0.5
    # # step_epoch = 3.0 #for Phase 1
    # step_epoch = 5.0 #for Phase 2
    # step_decay_weight = 0.9
    # step_decay = np.power(step_drop, np.floor((1. + epoch) / step_epoch))
    # sqrt_decay = 1./np.sqrt(1. + epoch)
    # return initial_lrate * ((step_decay_weight * step_decay) + ((1.0 - step_decay_weight) * sqrt_decay))

    #TODO Below Baseline 180906-1938
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    epoch -= 15 #fix for resuming from baseline Phase 2 DNN after Epoch 23
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
