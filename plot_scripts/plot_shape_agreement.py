#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator')
import cPickle as pickle
from utilities.generator import *

##################################################################################################

import argparse

parser = argparse.ArgumentParser(description='....', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-m', '--model', dest='folderMODEL', type=str, help='folderMODEL Path')
parser.add_argument('-w', '--weights', dest='num_weights', type=int, help='Load weights from Epoch')
parser.add_argument('-s', '--source', dest='source', type=str,
                    choices=['mixed', 'LB', 'bb0n', 'bb0nE', 'bb2n', 'gamma', 'Th228', 'Th232', 'U238', 'Xe135',
                             'Xe137', 'Bi214', 'Co60', 'Ra226', 'K40'], help='sources for training/validation')
parser.add_argument('-p', '--position', dest='position', type=str,
                    choices=['Uni', 'S2', 'S5', 'S8', 'S11', 'AllVessel', 'AllVessel-lowE', 'InnerCryo', 'AirGap',
                             'reduced'], help='source position')
parser.add_argument('-wp', '--wires', type=str, dest='wires', default='U', choices=['U', 'V', 'UV', 'U+V', 'small'], help='select wire planes')
args, unknown = parser.parse_known_args()

source, position, weights, wires = args.source, args.position, str(args.num_weights).zfill(3), args.wires

folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/%s/0validation/ShapeAgreement-%s-%s-%s-%s/'%(args.folderMODEL, weights, source, position, wires)
if not os.path.isdir(folderRUNS):
    os.makedirs(folderRUNS)
files = {}
files['1'] = (folderRUNS + '../%s-mc-%s-%s-%s/events_%s_%s-mc-%s-%s.hdf5')%(source, position, weights, wires, weights, source, position, wires)
files['2'] = (folderRUNS + '../%s-data-%s-%s-%s/events_%s_%s-data-%s-%s.hdf5')%(source, position, weights, wires, weights, source, position, wires)

for key in files.keys():
    print files[key]
    if not os.path.isfile(files[key]):
        raise OSError('file not found: %s'%files[key])

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1358/181023-1728/0validation/ShapeAgreement-067/'
# files = {}
# files['1'] = '../Th228-mc-S5-067-U/events_067_Th228-mc-S5-U.hdf5'
# files['2'] = '../Th228-data-S5-067-U/events_067_Th228-data-S5-U.hdf5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-bb2n-bb0nE/'
# files = {}
# files['1'] = '../bb2n-mc-Uni-023-U/events_023_bb2n-mc-Uni-U.hdf5'
# files['2'] = '../mixed-mc-Uni-023-U/events_023_mixed-mc-Uni-U.hdf5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Xe137-gamma/'
# files = {}
# files['1'] = '../Xe137-mc-Uni-023-U/events_023_Xe137-mc-Uni-U.hdf5'
# files['2'] = '../mixed-mc-Uni-023-U/events_023_mixed-mc-Uni-U.hdf5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Xe137-bb0nE/'
# files = {}
# files['1'] = '../Xe137-mc-Uni-023-U/events_023_Xe137-mc-Uni-U.hdf5'
# files['2'] = '../mixed-mc-Uni-023-U/events_023_mixed-mc-Uni-U.hdf5'

# TODO Baseline U+V DNN
# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181029-1015/0validation/ShapeAgreement-Th228/'
# files = {}
# files['1'] = '../Th228-mc-S5-020-UV/events_020_Th228-mc-S5-UV.hdf5'
# files['2'] = '../Th228-data-S5-020-UV/events_020_Th228-data-S5-UV.hdf5'
# source, position = 'Th228', 'S5'

# TODO Baseline U DNN
# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Th228-old/'
# files = {}
# files['1'] = '../Th228-mc-S5-023-U/events_023_Th228-mc-S5-U.hdf5'
# files['2'] = '../Th228-data-S5-023-U-old/events_023_Th228-data-S5-U.hdf5'
# source, position = 'Th228', 'S5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Th228-S5/'
# files = {}
# files['1'] = '../Th228-mc-S5-023-U/events_023_Th228-mc-S5-U.hdf5'
# files['2'] = '../Th228-data-S5-023-U/events_023_Th228-data-S5-U.hdf5'
# source, position = 'Th228', 'S5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Th228-S2/'
# files = {}
# files['1'] = '../Th228-mc-S2-023-U/events_023_Th228-mc-S2-U.hdf5'
# files['2'] = '../Th228-data-S2-023-U/events_023_Th228-data-S2-U.hdf5'
# source, position = 'Th228', 'S2'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Th228-S8/'
# files = {}
# files['1'] = '../Th228-mc-S8-023-U/events_023_Th228-mc-S8-U.hdf5'
# files['2'] = '../Th228-data-S8-023-U/events_023_Th228-data-S8-U.hdf5'
# source, position = 'Th228', 'S8'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Th228-S11/'
# files = {}
# files['1'] = '../Th228-mc-S11-023-U/events_023_Th228-mc-S11-U.hdf5'
# files['2'] = '../Th228-data-S11-023-U/events_023_Th228-data-S11-U.hdf5'
# source, position = 'Th228', 'S11'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Ra226-S5/'
# files = {}
# files['1'] = '../Ra226-mc-S5-023-U/events_023_Ra226-mc-S5-U.hdf5'
# files['2'] = '../Ra226-data-S5-023-U/events_023_Ra226-data-S5-U.hdf5'
# source, position = 'Ra226', 'S5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-Co60-S5/'
# files = {}
# files['1'] = '../Co60-mc-S5-023-U/events_023_Co60-mc-S5-U.hdf5'
# files['2'] = '../Co60-data-S5-023-U/events_023_Co60-data-S5-U.hdf5'
# source, position = 'Co60', 'S5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181030-1854/0validation/NERSC-bb0n/'
# files = {}
# files['1'] = '../bb0n-mc-reduced-045-small/events_045_bb0n-mc-reduced-small.hdf5'
# files['2'] = 'NERSC-file.hdf5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181030-1854/0validation/NERSC-Th232/'
# files = {}
# files['1'] = '../Th232-mc-reduced-045-small/events_045_Th232-mc-reduced-small.hdf5'
# files['2'] = 'NERSC-file.hdf5'

# TODO Baseline U DNN (AllVessel BKG)
# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1717/181023-1907/181024-1842/0validation/ShapeAgreement-Th228-S5/'
# files = {}
# files['1'] = '../Th228-mc-S5-025-U/events_025_Th228-mc-S5-U.hdf5'
# files['2'] = '../Th228-data-S5-025-U/events_025_Th228-data-S5-U.hdf5'
# source, position = 'Th228', 'S5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1717/181023-1907/181024-1842/0validation/ShapeAgreement-Th228-S8/'
# files = {}
# files['1'] = '../Th228-mc-S8-025-U/events_025_Th228-mc-S8-U.hdf5'
# files['2'] = '../Th228-data-S8-025-U/events_025_Th228-data-S8-U.hdf5'
# source, position = 'Th228', 'S8'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1717/181023-1907/181024-1842/0validation/ShapeAgreement-Th228-S11/'
# files = {}
# files['1'] = '../Th228-mc-S11-025-U/events_025_Th228-mc-S11-U.hdf5'
# files['2'] = '../Th228-data-S11-025-U/events_025_Th228-data-S11-U.hdf5'
# source, position = 'Th228', 'S11'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1717/181023-1907/181024-1842/0validation/ShapeAgreement-Th228-S2/'
# files = {}
# files['1'] = '../Th228-mc-S2-025-U/events_025_Th228-mc-S2-U.hdf5'
# files['2'] = '../Th228-data-S2-025-U/events_025_Th228-data-S2-U.hdf5'
# source, position = 'Th228', 'S2'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1717/181023-1907/181024-1842/0validation/ShapeAgreement-Ra226-S5/'
# files = {}
# files['1'] = '../Ra226-mc-S5-025-U/events_025_Ra226-mc-S5-U.hdf5'
# files['2'] = '../Ra226-data-S5-025-U/events_025_Ra226-data-S5-U.hdf5'
# source, position = 'Ra226', 'S5'

# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/181022-1717/181023-1907/181024-1842/0validation/ShapeAgreement-Co60-S5/'
# files = {}
# files['1'] = '../Co60-mc-S5-025-U/events_025_Co60-mc-S5-U.hdf5'
# files['2'] = '../Co60-data-S5-025-U/events_025_Co60-data-S5-U.hdf5'
# source, position = 'Co60', 'S5'

discriminator = 'signal-likeness'

def main():
    print 'starting'
    name_1 = 'MC'
    name_2 = 'Data'

    data = {}
    # for f in [os.path.join(files['2'], f) for f in os.listdir(files['2']) if os.path.isfile(os.path.join(files['2'], f))]:
    #     temp = read_hdf5_file_to_dict(f)
    #     for key in temp:
    #         if data == {}:
    #             data[key] = temp[key]
    #         else:
    #             data[key] = np.concatenate((data[key], temp[key]))
    # write_dict_to_hdf5_file(data=data, file=(folderOUT + 'NERSC-file.hdf5'))
    # exit()

    for key, model in files.items():
        data[key] = read_hdf5_file_to_dict(files[key])
        # data[key] = pickle.load(open(folderRUNS + files[key], "rb"))
        # files[key] = os.path.splitext(files[key])[0] + '.hdf5'
        # write_dict_to_hdf5_file(data=data[key], file=(folderRUNS + files[key]))

    if len(data['1']['DNNPredTrueClass'].shape) > 1:
        plots_multiOutput(data)
        return

    print 'Real data run:', set(data['2']['MCRunNumber'])

    mask1Fid = (np.sum(data['1']['CCIsFiducial'], axis=1) == data['1']['CCNumberClusters']) & (np.sum(data['1']['CCIs3DCluster'], axis=1) == data['1']['CCNumberClusters'])
    mask2Fid = (np.sum(data['2']['CCIsFiducial'], axis=1) == data['2']['CCNumberClusters']) & (np.sum(data['2']['CCIs3DCluster'], axis=1) == data['2']['CCNumberClusters'])
    for key in data['1'].keys():
        data['1'][key] = data['1'][key][mask1Fid]
    for key in data['2'].keys():
        data['2'][key] = data['2'][key][mask2Fid]

    mask1 = (data['1']['CCIsSS'] == 1) & (data['1']['DNNTrueClass'] == 0)
    mask2 = (data['2']['CCIsSS'] == 1) & (data['2']['DNNTrueClass'] == 0)

    rad1 = np.sqrt(data['1']['CCPosX'][:, 0] * data['1']['CCPosX'][:, 0] + data['1']['CCPosY'][:, 0] * data['1']['CCPosY'][:, 0])
    rad2 = np.sqrt(data['2']['CCPosX'][:, 0] * data['2']['CCPosX'][:, 0] + data['2']['CCPosY'][:, 0] * data['2']['CCPosY'][:, 0])

    print data['1'].keys()
    print data['1']['APDTime']
    print data['2']['APDTime']
    print data['1']['CCCollectionTime'][:,0]
    print data['2']['CCCollectionTime'][:,0]

    # plot_hist2_multi(np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1], data['1']['APDTime'][mask1],
    #                  np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2], data['2']['APDTime'][mask2],
    #                  [1000, 3000], [900, 1050], 'Energy', 'APD Time', name_1, name_2, 'Energy_vs_APD-Time_SS.pdf')
    #
    # plot_hist2_multi(np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1], data['1']['CCCollectionTime'][:,0][mask1],
    #                  np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2], data['2']['CCCollectionTime'][:,0][mask2],
    #                  [1000, 3000], [1000, 1150], 'Energy', 'CC Time', name_1, name_2, 'Energy_vs_CC-Time_SS.pdf')

    kwargs = {
        'range': (0, 1),
        'bins': 50,
        'density': False
    }

    e_limit = 1000. #2000.
    maskE1 = np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1) > e_limit
    maskE2 = np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1) > e_limit
    t_limit = 1025
    maskT1 = data['1']['CCCollectionTime'][:, 0] > t_limit
    maskT2 = data['2']['CCCollectionTime'][:, 0] > t_limit
    # t_limit = 1010
    # maskT1 = data['1']['APDTime'] > t_limit
    # maskT2 = data['2']['APDTime'] > t_limit

    make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask1 & maskT1],
                              data['2']['DNNPredTrueClass'][mask2 & maskT2],
                              'SS', '%s (T gr %d)' % (discriminator, t_limit), **kwargs)
    make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask1 & ~maskE1 & maskT1],
                              data['2']['DNNPredTrueClass'][mask2 & ~maskE2 & maskT2],
                              'SS', '%s (E le %d) (T gr %d)' % (discriminator, e_limit, t_limit), **kwargs)
    make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask1 & ~maskE1 & ~maskT1],
                              data['2']['DNNPredTrueClass'][mask2 & ~maskE2 & ~maskT2],
                              'SS', '%s (E le %d) (T le %d)' % (discriminator, e_limit, t_limit), **kwargs)

    kwargs = {
            'range': (1016, 1026),
        'bins': 150,
        'density': False
    }
    make_shape_agreement_plot(data['1']['APDTime'][mask1],
                              data['2']['APDTime'][mask2],
                              'SS', 'APD_Time', **kwargs)

    kwargs = {
            'range': (1024, 1026),
        'bins': 50,
        'density': False
    }
    make_shape_agreement_plot(data['1']['APDTime'][mask1],
                              data['1']['APDTime'][mask1],
                              'SS', 'APD_Time_MC', **kwargs)

    print 'mean:', np.mean(data['1']['APDTime'][mask1])
    print 'std:', np.std(data['1']['APDTime'][mask1])

    kwargs = {
            'range': (0, 120), # (0, 110) for Phase-2
        'bins': 50,
        'density': False
    }
    make_shape_agreement_plot((data['1']['CCCollectionTime'][:, 0]-data['1']['APDTime'])[mask1],
                              (data['2']['CCCollectionTime'][:, 0]-data['2']['APDTime'])[mask2],
                              'SS', 'CC_TimeDiff', **kwargs)
    # exit()
    kwargs = {
        'range': (1020, 1130),
        'bins': 50,
        'density': False
    }

    make_shape_agreement_plot(data['1']['CCCollectionTime'][:, 0][mask1 & maskT1],
                              data['2']['CCCollectionTime'][:, 0][mask2 & maskT2],
                              'SS', 'CC_Time_Tgr1025', **kwargs)

    make_shape_agreement_plot(data['1']['CCCollectionTime'][:, 0][mask1],
                              data['2']['CCCollectionTime'][:, 0][mask2],
                              'SS', 'CC_Time', **kwargs)

    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1 & maskT1], np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1 & maskT1],
                     data['2']['DNNPredTrueClass'][mask2 & maskT2], np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2 & maskT2],
                     [0.0, 1.0], [1000, 3000], discriminator, 'Energy', name_1, name_2,
                     '%s_vs_Energy_Tgr1025_SS.pdf' % (discriminator))

    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1 & maskT1], np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1 & maskT1],
                          data['2']['DNNPredTrueClass'][mask2 & maskT2], np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2 & maskT2],
                          [0.0, 1.0], [1000, 2400], discriminator, 'Energy', name_1, name_2,
                          '%s_vs_Energy_Tgr1025_SS_norm.pdf' % (discriminator))

    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1 & maskT1], data['1']['CCCollectionTime'][:,0][mask1 & maskT1],
                     data['2']['DNNPredTrueClass'][mask2 & maskT2], data['2']['CCCollectionTime'][:,0][mask2 & maskT2],
                     [0.0, 1.0], [1000, 1150], discriminator, 'CCTime', name_1, name_2,
                     '%s_vs_CCTime_Tgr1025_SS.pdf' % (discriminator))

    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1 & maskT1], data['1']['CCCollectionTime'][:,0][mask1 & maskT1],
                          data['2']['DNNPredTrueClass'][mask2 & maskT2], data['2']['CCCollectionTime'][:,0][mask2 & maskT2],
                          [0.0, 1.0], [1000, 1150], discriminator, 'CCTime', name_1, name_2,
                          '%s_vs_CCTime_Tgr1025_SS_norm.pdf' % (discriminator))

    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCCollectionTime'][:,0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCCollectionTime'][:,0][mask2],
                     [0.0, 1.0], [1000, 1150], discriminator, 'CCTime', name_1, name_2,
                     '%s_vs_CCTime_SS.pdf' % (discriminator))

    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCCollectionTime'][:,0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCCollectionTime'][:,0][mask2],
                          [0.0, 1.0], [1000, 1150], discriminator, 'CCTime', name_1, name_2,
                          '%s_vs_CCTime_SS_norm.pdf' % (discriminator))

    # exit()

    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosU'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosU'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'U', name_1, name_2, '%s_vs_U_SS.pdf' % (discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosV'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosV'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'V', name_1, name_2, '%s_vs_V_SS.pdf' % (discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], rad1[mask1],
                     data['2']['DNNPredTrueClass'][mask2], rad2[mask2],
                     [0.0, 1.0], [0, 180], discriminator, 'R', name_1, name_2, '%s_vs_R_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosX'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosX'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'X', name_1, name_2, '%s_vs_X_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosY'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosY'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'Y', name_1, name_2, '%s_vs_Y_SS.pdf'%(discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosZ'][:, 0][mask1],
                     data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosZ'][:, 0][mask2],
                     [0.0, 1.0], [-200, 200], discriminator, 'Z', name_1, name_2, '%s_vs_Z_SS.pdf'%(discriminator))
    # plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], np.sum(data['1']['CCCorrectedEnergy'], axis=1)[mask1],
    #                  data['2']['DNNPredTrueClass'][mask2], np.sum(data['2']['CCCorrectedEnergy'], axis=1)[mask2],
    #                  [0.0, 1.0], [1000, 3000], discriminator, 'Energy', name_1, name_2, '%s_vs_Energy_SS.pdf' % (discriminator))
    plot_hist2_multi(data['1']['DNNPredTrueClass'][mask1], np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1],
                     data['2']['DNNPredTrueClass'][mask2], np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2],
                     [0.0, 1.0], [1000, 3000], discriminator, 'Energy', name_1, name_2, '%s_vs_Energy_SS.pdf'%(discriminator))


    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], rad1[mask1],
                          data['2']['DNNPredTrueClass'][mask2], rad2[mask2],
                          [0.0, 1.0], [0, 180], discriminator, 'R', name_1, name_2, '%s_vs_R_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosX'][:, 0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosX'][:, 0][mask2],
                          [0.0, 1.0], [-200, 200], discriminator, 'X', name_1, name_2, '%s_vs_X_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosY'][:, 0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosY'][:, 0][mask2],
                          [0.0, 1.0], [-200, 200], discriminator, 'Y', name_1, name_2, '%s_vs_Y_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], data['1']['CCPosZ'][:, 0][mask1],
                          data['2']['DNNPredTrueClass'][mask2], data['2']['CCPosZ'][:, 0][mask2],
                          [0.0, 1.0], [-200, 200], discriminator, 'Z', name_1, name_2, '%s_vs_Z_SS_norm.pdf'%(discriminator))
    # plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], np.sum(data['1']['CCCorrectedEnergy'], axis=1)[mask1],
    #                       data['2']['DNNPredTrueClass'][mask2], np.sum(data['2']['CCCorrectedEnergy'], axis=1)[mask2],
    #                       [0.0, 1.0], [1000, 3000], discriminator, 'Energy', name_1, name_2, '%s_vs_Energy_SS_norm.pdf'%(discriminator))
    plot_hist2_multi_norm(data['1']['DNNPredTrueClass'][mask1], np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1)[mask1],
                          data['2']['DNNPredTrueClass'][mask2], np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1)[mask2],
                          [0.0, 1.0], [1000, 2400], discriminator, 'Energy', name_1, name_2, '%s_vs_Energy_SS_norm.pdf' % (discriminator))

    # exit()

    mask_1y = data['1']['DNNTrueClass'] == 0
    mask_2y = data['2']['DNNTrueClass'] == 0

    for i in xrange(3):
        if i == 0:
            continue
            mask1 = mask2 = True
            title = 'SS+MS'
        elif i == 1:
            mask1 = (data['1']['CCIsSS'] == 1)
            mask2 = (data['2']['CCIsSS'] == 1)
            title = 'SS'
        elif i == 2:
            # continue
            mask1 = (data['1']['CCIsSS'] == 0)
            mask2 = (data['2']['CCIsSS'] == 0)
            title = 'MS'
        else:
            raise ValueError('check loop')

        rad1 = np.sqrt(data['1']['CCPosX'][:, 0] * data['1']['CCPosX'][:, 0] + data['1']['CCPosY'][:, 0] * data['1']['CCPosY'][:, 0])
        rad2 = np.sqrt(data['2']['CCPosX'][:, 0] * data['2']['CCPosX'][:, 0] + data['2']['CCPosY'][:, 0] * data['2']['CCPosY'][:, 0])

        kwargs = {
            'range': (500, 3000),
            'bins': 50,
            'density': False
        }

        make_shape_agreement_plot(np.sum(data['1']['CCPurityCorrectedEnergy'][mask_1y & mask1], axis=1),
                                  np.sum(data['2']['CCPurityCorrectedEnergy'][mask_2y & mask2], axis=1), title, 'energy', **kwargs)

        kwargs = {
            'range': (0, 1),
            'bins': 50,
            'density': False
        }

        e_limit = 1000. #2000.
        maskE1 = np.sum(data['1']['CCPurityCorrectedEnergy'], axis=1) > e_limit
        maskE2 = np.sum(data['2']['CCPurityCorrectedEnergy'], axis=1) > e_limit
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskE1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskE2] , title, '%s (E gr %d)' % (discriminator, e_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~maskE1], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~maskE2], title, '%s (E le %d)' % (discriminator, e_limit), **kwargs)


        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1], data['2']['DNNPredTrueClass'][mask_2y & mask2], title, discriminator, **kwargs)

        rad_limit = 160.
        maskR1 = rad1 < rad_limit
        maskR2 = rad2 < rad_limit
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskR1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskR2] , title, '%s (R le %d)'%(discriminator, rad_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~maskR1], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~maskR2], title, '%s (R gr %d)'%(discriminator, rad_limit), **kwargs)

        z_limit = 20.
        maskZ1 = np.abs(data['1']['CCPosZ'][:, 0]) > z_limit
        maskZ2 = np.abs(data['2']['CCPosZ'][:, 0]) > z_limit
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskZ1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskZ2] , title, '%s (Z gr %d)'%(discriminator, z_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~maskZ1], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~maskZ2], title, '%s (Z le %d)'%(discriminator, z_limit), **kwargs)


        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskZ1 & maskR1] , data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskZ2 & maskR2] , title, '%s (Z gr %d + R le %d)'%(discriminator, z_limit, rad_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~(maskZ1 & maskR1)], data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~(maskZ2 & maskR2)], title, '%s not(Z gr %d + R le %d)'%(discriminator, z_limit, rad_limit), **kwargs)

        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & maskZ1 & maskR1 & maskE1],
                                  data['2']['DNNPredTrueClass'][mask_2y & mask2 & maskZ2 & maskR2 & maskE2], title,
                                  '%s (Z gr %d + R le %d + E gr %d)' % (discriminator, z_limit, rad_limit, e_limit), **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask_1y & mask1 & ~(maskZ1 & maskR1 & maskE1)],
                                  data['2']['DNNPredTrueClass'][mask_2y & mask2 & ~(maskZ2 & maskR2 & maskE2)], title,
                                  '%s not(Z gr %d + R le %d + E gr %d)' % (discriminator, z_limit, rad_limit, e_limit), **kwargs)

        # kwargs = {
        #     'range': (-180, 180),
        #     'bins': 50,
        #     'density': False
        # }
        # make_shape_agreement_plot(data['1'], data['2'], data['1']['CCPosZ'][:,0], data['2']['CCPosZ'][:,0], 'Z', **kwargs)
        #
        # kwargs = {
        #     'range': (0, 175),
        #     'bins': 50,
        #     'density': False
        # }
        # make_shape_agreement_plot(data['1'], data['2'], rad1, rad2, 'radius', **kwargs)


def plots_multiOutput(data):
    mask_1y = data['1']['DNNTrueClass'][:, 0] == 0
    mask_2y = data['2']['DNNTrueClass'][:, 0] == 0

    kwargs = {
        'range': (0, 1),
        'bins': 50,
        'density': False
    }

    for i in xrange(3):
        if i == 0:
            continue
            mask1 = mask2 = True
            title = 'SS+MS'
        elif i == 1:
            mask1 = (data['1']['CCIsSS'] == 1)
            mask2 = (data['2']['CCIsSS'] == 1)
            title = 'SS'
        elif i == 2:
            mask1 = (data['1']['CCIsSS'] == 0)
            mask2 = (data['2']['CCIsSS'] == 0)
            title = 'MS'
        else:
            raise ValueError('check loop')

        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][:, 0][mask_1y & mask1],
                                  data['2']['DNNPredTrueClass'][:, 0][mask_2y & mask2],
                                  title, discriminator+'-Final', **kwargs)
        make_shape_agreement_plot(data['1']['DNNPredTrueClass'][:, 1][mask_1y & mask1],
                                  data['2']['DNNPredTrueClass'][:, 1][mask_2y & mask2],
                                  title, discriminator+'-Top', **kwargs)


def make_shape_agreement_plot(mc, data, title, name, **kwargs):
    hist_1y, bin_edges = np.histogram(mc, **kwargs)
    hist_2y, bin_edges = np.histogram(data, **kwargs)

    hist_2y_err = np.sqrt(hist_2y)

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    nevents1 = float(sum(hist_1y))
    nevents2 = float(sum(hist_2y))
    binwidth = (bin_edges[1] - bin_edges[0])
    hist_1y = hist_1y / nevents1 / binwidth
    hist_2y = hist_2y / nevents2 / binwidth
    hist_2y_err = hist_2y_err / nevents2 / binwidth

    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex = ax1)
    ax1.step(bin_centres, hist_1y, where='mid', color='blue', label='MC (%s)'%(title))
    ax1.errorbar(bin_centres, hist_2y, hist_2y_err, color='k', fmt='.', label='%s (%s)'%(source, position))
    ax2.axhline(y=0., c='k')
    # ax2.axhline(y=+0.25, c='k', alpha=0.5)
    # ax2.axhline(y=-0.25, c='k', alpha=0.5)
    ax2.errorbar(bin_centres, (hist_2y-hist_1y)/hist_1y, hist_2y_err/hist_1y, color='k', fmt='.', label='%s (%s)'%(source, position))
    # ax2.scatter(bin_centres, (hist_2y-hist_1y)/hist_2y_err, color='k')
    ax2.set_xlabel(name)
    ax2.set_ylabel('(data-MC)/MC')
    ax1.legend(loc='upper center')
    # ax1.set_title(title)
    ax1.set_xlim(kwargs['range'])
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(-0.4, 0.4)
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.0)
    f.savefig(folderRUNS + name+ '_' + title + '.pdf', bbox_inches='tight')
    plt.close()

#TODO Check difference plot ax3. Looks strange
def plot_hist2_multi_norm(E_x1, E_y1, E_x2, E_y2, range_x, range_y, name_x, name_y, name_1, name_2, fOUT):
    pos_bins = 25

    hist1D_Z1, bin_edges = np.histogram(E_y1, range=range_y, bins=pos_bins, normed=True)
    hist1D_Z2, bin_edges = np.histogram(E_y2, range=range_y, bins=pos_bins, normed=True)

    weights_Z1 = np.asarray([1. / hist1D_Z1[np.argmax(bin_edges >= p) - 1]
                             if p >= bin_edges[0] and p < bin_edges[-1] else 0.0 for p in E_y1])
    weights_Z2 = np.asarray([1. / hist1D_Z2[np.argmax(bin_edges >= p) - 1]
                             if p >= bin_edges[0] and p < bin_edges[-1] else 0.0 for p in E_y2])

    hist2D_Z1, xbins, ybins = np.histogram2d(E_x1, E_y1, weights=weights_Z1, range=[range_x, range_y],
                                            bins=pos_bins, normed=True)
    hist2D_Z2, xbins, ybins = np.histogram2d(E_x2, E_y2, weights=weights_Z2, range=[range_x, range_y],
                                            bins=pos_bins, normed=True)

    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = "auto"

    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    max12 = 100. * np.max([hist2D_Z1, hist2D_Z2])
    h2 = ax2.imshow(100.*np.ma.masked_where(hist2D_Z2 == 0 , hist2D_Z2).T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                   aspect=aspect, norm=colors.Normalize(vmax=max12))
    f.colorbar(h2, ax=ax2, shrink=0.8)
    h1 = ax1.imshow(100.*np.ma.masked_where(hist2D_Z1 == 0 , hist2D_Z1).T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=colors.Normalize(vmax=max12))
    f.colorbar(h1, ax=ax1, shrink=0.8)

    hist2D_Z3 = (hist1D_Z1 - hist2D_Z2)
    max3 = 100.*np.max(np.abs(hist2D_Z3))
    h3 = ax3.imshow(100.*np.ma.masked_where(hist2D_Z3 == 0 , hist2D_Z3).T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('RdBu_r'), origin='lower',
                    aspect=aspect, norm=colors.Normalize(vmin=-max3, vmax=max3))
    f.colorbar(h3, ax=ax3, shrink=0.8)

    ax3.set_xlabel(name_x)
    ax1.set_ylabel('%s (%s)'%(name_y, name_1))
    ax2.set_ylabel('%s (%s)'%(name_y, name_2))
    ax3.set_ylabel('%s (%s-%s)'%(name_y, name_1, name_2))
    ax1.set_xlim(range_x)
    ax1.set_ylim(range_y)
    ax2.set_ylim(range_y)
    ax3.set_ylim(range_y)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    f.savefig(folderRUNS+fOUT, bbox_inches='tight')
    plt.close()


def plot_hist2_multi(E_x1, E_y1, E_x2, E_y2, range_x, range_y, name_x, name_y, name_1, name_2, fOUT):
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)

    numEv = min((E_x1.shape[0],E_x2.shape[0]))
    # E_y1 = np.abs(E_y1)
    # E_y2 = np.abs(E_y2)

    h1 = ax1.hexbin(E_x1[:numEv], E_y1[:numEv], extent=extent, gridsize=25, linewidths=0.1, norm=colors.Normalize(vmax=120), cmap=plt.get_cmap('viridis'))
    f.colorbar(h1, ax=ax1, shrink=0.6)
    h2 = ax2.hexbin(E_x2[:numEv], E_y2[:numEv], extent=extent, gridsize=25, linewidths=0.1, norm=colors.Normalize(vmax=120), cmap=plt.get_cmap('viridis'))
    f.colorbar(h2, ax=ax2, shrink=0.6)

    # max3 = np.max(h2.get_array() - h1.get_array())
    h3 = ax3.hexbin(E_x2[:numEv], E_y2[:numEv], extent=extent, gridsize=25, linewidths=0.1, vmin=-50, vmax=50, cmap=plt.get_cmap('RdBu_r'))
    h3.set_array(h1.get_array() - h2.get_array())
    f.colorbar(h3, ax=ax3, shrink=0.6)

    ax3.set_xlabel(name_x)
    ax1.set_ylabel('%s (%s)'%(name_y, name_1))
    ax2.set_ylabel('%s (%s)'%(name_y, name_2))
    ax3.set_ylabel('%s (%s-%s)'%(name_y, name_1, name_2))
    ax1.set_xlim(range_x)
    ax1.set_ylim(range_y)
    ax2.set_ylim(range_y)
    ax3.set_ylim(range_y)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    # plt.subplots_adjust(hspace=.0)
    f.savefig(folderRUNS+fOUT, bbox_inches='tight')
    plt.close()


# scatter 2D heatmap
def plot_scatter_hist2d(E_x, E_y, range_x, range_y, name_x, name_y, name_title, fOUT):
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]
    plt.hexbin(E_x, E_y, extent=extent, gridsize=50, mincnt=1, linewidths=0.1, cmap=plt.get_cmap('viridis'))

    plt.title('%s' % (name_title))
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.xlim(xmin=range_x[0], xmax=range_x[1])
    plt.ylim(ymin=range_y[0], ymax=range_y[1])
    plt.grid(True)
    plt.savefig(folderRUNS+fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

if __name__ == '__main__':

    main()

    print '===================================== Program finished =============================='
