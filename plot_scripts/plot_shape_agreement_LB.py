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

# TODO Baseline U+V DNN
folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/ShapeAgreement-LB/'
files = {}
files['LB'] = '../LB-data-AllVessel-023-U/events_023_LB-data-AllVessel-U.hdf5'
files['bb2n'] = '../bb2n-mc-Uni-023-U/events_023_bb2n-mc-Uni-U.hdf5'
files['Vessel_K40'] = '../K40-mc-AllVessel-023-U/events_023_K40-mc-AllVessel-U.hdf5'
files['Vessel_Th232'] = '../Th232-mc-AllVessel-lowE-023-U/events_023_Th232-mc-AllVessel-lowE-U.hdf5'
files['Vessel_U238'] = '../U238-mc-AllVessel-lowE-023-U/events_023_U238-mc-AllVessel-lowE-U.hdf5'
files['Vessel_Co60'] = '../Co60-mc-AllVessel-023-U/events_023_Co60-mc-AllVessel-U.hdf5'
files['Cryo_Th232'] = '../Th232-mc-InnerCryo-023-U/events_023_Th232-mc-InnerCryo-U.hdf5'
files['LXe_Xe137'] = '../Xe137-mc-Uni-023-U/events_023_Xe137-mc-Uni-U.hdf5'
files['LXe_Xe135'] = '../Xe135-mc-Uni-023-U/events_023_Xe135-mc-Uni-U.hdf5'
files['AirGap_214_Bi'] = '../Bi214-mc-AirGap-023-U/events_023_Bi214-mc-AirGap-U.hdf5'

names = {}
names['bb2n'] = 'bb2n'
names['Vessel_K40'] = 'AV-K40'
names['Vessel_Th232'] = 'AV-Th232'
names['Vessel_U238'] = 'AV-U238'
names['Vessel_Co60'] = 'AV-Co60'
names['Cryo_Th232'] = 'IC-Th232'
names['LXe_Xe137'] = 'Xe137'
names['LXe_Xe135'] = 'Xe135'
names['AirGap_214_Bi'] = 'AG-Bi214'

discriminator = 'signal-likeness'
mult = 'MS'
# mult = 'SS'

folderDB = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Plots/'
filesDB = [ os.path.join(folderDB, f) for f in os.listdir(folderDB) if os.path.isfile(os.path.join(folderDB, f)) and 'LB_weights' in f and mult in f ]

def main():
    print 'starting'
    data = {}
    mask = {}
    for key in files.keys():
        data[key] = read_hdf5_file_to_dict(folderRUNS + files[key])
        if mult == 'SS': mask[key] = data[key]['CCIsSS'] == 1
        elif mult == 'MS': mask[key] = data[key]['CCIsSS'] == 0
        else: raise ValueError('multiplicity strange')

    name_1 = 'MC'
    name_2 = 'Data'

    for file in filesDB:
        print '\n\nfile:\t', os.path.basename(file)
        f = open(file, 'r')
        f_lines = f.readlines()
        f.close()
        pdfs = {}
        for line in f_lines:
            pdf = line.split()
            pdfs[pdf[1]] = float(pdf[2])
            e_limit_temp = pdf[0].split("<")
            if len(e_limit_temp) != 3: raise ValueError('e limit shape is strange')
            e_limit = []
            if e_limit != []: continue
            for i in [0, 2]:
                if 'MeV' in e_limit_temp[i]: e_limit.append(float(e_limit_temp[i].split("MeV")[0]) * 1.e3)
                elif 'keV' in e_limit_temp[i]: e_limit.append(float(e_limit_temp[i].split("MeV")[0]) * 1.e0)
                else: raise ValueError('e limit shape is strange')

        print "energy limits:", e_limit
        if e_limit[0] < 2000:
            kwargs = {'range': (0, 1), 'bins': 50}
        else:
            kwargs = {'range': (0, 1), 'bins': 20}
        # if e_limit[0] < 2000:
        #     kwargs = {'range': (e_limit[0], e_limit[1]), 'bins': 40}
        # else:
        #     kwargs = {'range': (e_limit[0], e_limit[1]), 'bins': 20}

        maskE = {}
        data_temp = []
        weights_temp = []
        key_temp = []
        for key in files.keys():
            maskE[key] = (np.sum(data[key]['CCPurityCorrectedEnergy'], axis=1) > e_limit[0]) & \
                         (np.sum(data[key]['CCPurityCorrectedEnergy'], axis=1) < e_limit[1])
            if key != 'LB':
                key_temp.append(names[key])
                data_temp.append(data[key]['DNNPredTrueClass'][mask[key] & maskE[key]])
                # data_temp.append(np.sum(data[key]['CCPurityCorrectedEnergy'], axis=1)[mask[key] & maskE[key]])
                pdf = filter(lambda x: key in x, pdfs.keys())
                print key, pdf, [ pdfs[i] for i in pdf ]
                if len(pdf) != 1: raise Exception('pdf not found or not clear: %s'%(key))
                weights_temp.append(pdfs[pdf[0]])

        make_shape_agreement_plot_combined(data['LB']['DNNPredTrueClass'][mask['LB'] & maskE['LB']],
                                           data_temp, weights_temp, key_temp,
                                           '%s_%s-%s'%(mult, str(int(e_limit[0])), str(int(e_limit[1]))), str(discriminator),
                                           log=False, **kwargs)

        make_shape_agreement_plot_combined(data['LB']['DNNPredTrueClass'][mask['LB'] & maskE['LB']],
                                           data_temp, weights_temp, key_temp,
                                           '%s_%s-%s-log'%(mult, str(int(e_limit[0])), str(int(e_limit[1]))), str(discriminator),
                                           log=True, **kwargs)

        # make_shape_agreement_plot_combined(np.sum(data['LB']['CCPurityCorrectedEnergy'], axis=1)[mask['LB'] & maskE['LB']],
        #                                    data_temp, weights_temp, key_temp,
        #                                    '%s_%s-%s' % (mult, str(int(e_limit[0])), str(int(e_limit[1]))),
        #                                    'Energy', log=False, **kwargs)
        #
        # make_shape_agreement_plot_combined(np.sum(data['LB']['CCPurityCorrectedEnergy'], axis=1)[mask['LB'] & maskE['LB']],
        #                                    data_temp, weights_temp, key_temp,
        #                                    '%s_%s-%s-log' % (mult, str(int(e_limit[0])), str(int(e_limit[1]))),
        #                                    'Energy', log=True, **kwargs)

    exit()


    make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask1 & ~maskE1 & maskT1],
                              data['2']['DNNPredTrueClass'][mask2 & ~maskE2 & maskT2],
                              'SS', '%s (E le %d) (T gr %d)' % (discriminator, e_limit, t_limit), **kwargs)
    make_shape_agreement_plot(data['1']['DNNPredTrueClass'][mask1 & ~maskE1 & ~maskT1],
                              data['2']['DNNPredTrueClass'][mask2 & ~maskE2 & ~maskT2],
                              'SS', '%s (E le %d) (T le %d)' % (discriminator, e_limit, t_limit), **kwargs)

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
            'range': (1000, 3000),
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

        e_limit = 1300. #2000.
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




def make_shape_agreement_plot_combined(data, mc, weights, title, name, xlabel, log=False, **kwargs):
    if not isinstance(mc, list):
        raise TypeError('passed MC variable need to be list')

    hist_data, bin_edges = np.histogram(data, density=False, **kwargs)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    binwidth = (bin_edges[1] - bin_edges[0])
    hist_mc = []
    hist_mc_err = []
    hist_mc_all = np.zeros(hist_data.shape)
    for i in range(len(mc)):
        hist_temp, bin_edges = np.histogram(mc[i], density=False, **kwargs)
        hist_mc.append(hist_temp / float(sum(hist_temp)) / binwidth)
        hist_mc_err.append(np.sqrt(hist_temp) / float(sum(hist_temp)) / binwidth)
        if mc[i].size == 0:
            continue
        hist_mc_all += hist_temp / float(sum(hist_temp)) / binwidth * weights[i]

    nevents_data = float(sum(hist_data))
    hist_data_err = np.sqrt(hist_data) / nevents_data / binwidth
    hist_data = hist_data / nevents_data / binwidth

    all_weights = float(sum(weights))
    # hist_mc_all_err = np.sqrt(hist_mc_all) / all_weights
    hist_mc_all /= all_weights

    plt.clf()
    f = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex = ax1)
    for i in range(len(mc)):
        if weights[i] == 0.0: continue
        ax1.errorbar(bin_centres, hist_mc[i]*weights[i]/all_weights, hist_mc_err[i]*weights[i]/all_weights, c='C%i'%(i), fmt='none')
        ax1.step(bin_centres, hist_mc[i]*weights[i]/all_weights, where='mid', c='C%i'%(i), label='%s'%(title[i]))
    # ax1.errorbar(bin_centres, hist_mc_all, hist_mc_all_err, color='blue', fmt='none')
    ax1.step(bin_centres, hist_mc_all, where='mid', color='blue', label='MC all')
    ax1.errorbar(bin_centres, hist_data, hist_data_err, color='k', fmt='.', label='LB data')
    ax2.axhline(y=0., c='k')
    ax2.errorbar(bin_centres, (hist_data-hist_mc_all)/hist_mc_all, hist_data_err/hist_mc_all, color='k', fmt='.')

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('(data-MC)/MC')
    # ax1.legend(loc='best', ncol=2)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    if log: ax1.set_yscale("log")
    ax1.set_xlim(kwargs['range'])
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(-0.7, 0.7)
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.0)
    f.savefig(folderRUNS + xlabel + '_' + name + '.pdf', bbox_inches='tight')
    plt.close()

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
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex = ax1)
    ax1.step(bin_centres, hist_1y, where='mid', color='blue', label='MC (%s)'%(title))
    ax1.errorbar(bin_centres, hist_2y, hist_2y_err, color='k', fmt='.', label='Th228 (S5)')
    ax2.axhline(y=0., c='k')
    # ax2.axhline(y=+0.25, c='k', alpha=0.5)
    # ax2.axhline(y=-0.25, c='k', alpha=0.5)
    ax2.errorbar(bin_centres, (hist_2y-hist_1y)/hist_1y, hist_2y_err/hist_1y, color='k', fmt='.', label='Th228 (S5)')
    # ax2.scatter(bin_centres, (hist_2y-hist_1y)/hist_2y_err, color='k')
    ax2.set_xlabel(name)
    ax2.set_ylabel('(data-MC)/MC')
    ax1.legend(loc='upper center')
    # ax1.set_title(title)
    ax1.set_xlim(kwargs['range'])
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(-0.7, 0.7)
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
