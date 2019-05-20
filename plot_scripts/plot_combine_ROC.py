#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator')
import cPickle as pickle
from utilities.generator import *
from plot_scripts.plot_validation import *

# scatter
def plot_scatter(E_x, E_y, name_x, name_y, name_title, fOUT):
    plt.scatter(E_x, E_y)
    plt.plot([0,1], [0,1], 'k--')
    # plt.legend(loc="best")
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# scatter 2D heatmap
def plot_scatter_hist2d(E_x, E_y, name_x, name_y, name_title, fOUT):
    hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[[0,1],[0,1]], bins=50, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    plt.plot([0,1], [0,1], 'k-', lw=0.3)
    # im = plt.imshow(hist.T, extent=extent, interpolation='nearest', vmin=0, vmax=1, cmap=plt.get_cmap('viridis'), origin='lower', norm=mpl.colors.Normalize()) # norm=mpl.colors.LogNorm())
    plt.hexbin(E_x, E_y, extent=extent, gridsize=70, mincnt=1, lw =0.1, cmap=plt.get_cmap('viridis'))  # norm=mpl.colors.LogNorm())
    plt.title('%s' % (name_title))
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# residual (Hist2D)
def plot_residual_hist2d(E_x, E_y, name_x, name_y, name_title, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, dE, range=[[0, 1], [-1, 1]], bins=100, normed=True)
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    plt.axhline(y=0., color='k')
    im = plt.hexbin(E_x, dE, extent=extent, gridsize=100, mincnt=1, lw=0.1, cmap=plt.get_cmap('viridis'))

    # cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    # cbar.set_label('Probability')
    plt.title('%s' % (name_title))
    plt.xlabel('%s Energy' % (name_x))
    plt.ylabel('Residual (%s - %s)' % (name_y, name_x))
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=-0.2, ymax=0.2)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

def make_shape_agreement_plot(mc, data, label_mc, label_data, name, **kwargs):
    if isinstance(mc, list):
        pass
    elif isinstance(mc, np.ndarray):
        mc = [mc]
        data = [data]
        label_mc = [label_mc]
        label_data = [label_data]

    plt.clf()
    f = plt.figure() #figsize=(6,6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    for i in xrange(len(mc)):
        hist_1y, bin_edges = np.histogram(mc[i], **kwargs)
        hist_2y, bin_edges = np.histogram(data[i], **kwargs)
        hist_2y_err = np.sqrt(hist_2y)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        nevents1 = float(sum(hist_1y))
        nevents2 = float(sum(hist_2y))
        binwidth = (bin_edges[1] - bin_edges[0])
        hist_1y = hist_1y / nevents1 / binwidth
        hist_2y = hist_2y / nevents2 / binwidth
        hist_2y_err = hist_2y_err / nevents2 / binwidth

        ax1.step(bin_centres, hist_1y, where='mid', color='C%i'%i, label=label_mc[i])
        ax1.errorbar(bin_centres, hist_2y, hist_2y_err, color='C%i'%i, fmt='.', label=label_data[i])
        ax2.axhline(y=0., c='k')
        ax2.errorbar(bin_centres, (hist_2y - hist_1y) / hist_1y, hist_2y_err / hist_1y, color='C%i'%i, ls='-', fmt='.', label='')
    ax2.set_xlabel('signal-likeness')
    # ax2.set_ylabel('(data-MC)/MC')
    ax2.set_ylabel('(w/ - w/o) / w/o')
    ax1.legend(loc='upper center')
    ax1.set_xlim(kwargs['range'])
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(-0.25, 0.25)
    plt.setp(ax1.get_xticklabels(), visible=False)
    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.subplots_adjust(hspace=.1)
    f.savefig(folderRUNS + name + '.pdf', bbox_inches='tight')
    plt.close()

def GetData(files, kCombined=False):
    if not kCombined:
        data = {}
        for key in files.keys():
            data[key] = read_hdf5_file_to_dict(folderRUNS + files[key])
            print data[key].keys()
            print data[key].values()[0].shape[0]
        return  data


    if False and os.path.isfile(folderRUNS + 'combined.hdf5'):
        print "Loading combined file"
        data = read_hdf5_file_to_dict(folderRUNS + 'combined.hdf5')
        return data
    else:
        print "Remake the pickle file"

    keyLow, keyHigh = None, None
    numLow, numHigh = np.inf, 0
    data = {}
    for key in files.keys():
        data[key] = read_hdf5_file_to_dict(folderRUNS + files[key])
        # print data[key].keys()
        # print data[key].values()[0].shape[0]
        # if data[key].values()[0].shape[0] > numHigh:
        #     keyHigh = key
        #     numHigh = data[key].values()[0].shape[0]
        # if data[key].values()[0].shape[0] < numLow:
        #     keyLow = key
        #     numLow = data[key].values()[0].shape[0]

    keyLow = 'w/ Ind'
    keyHigh = 'w/o Ind'

    data[keyLow]['DNNPredTrueClassNoInd'] = np.zeros(data[keyLow]['DNNPredTrueClass'].shape, dtype=np.float32)
    for i in xrange(data[keyLow].values()[0].shape[0]):
        index = np.where((data[keyLow]['MCRunNumber'][i] == data[keyHigh]['MCRunNumber']) &
                         (data[keyLow]['MCEventNumber'][i] == data[keyHigh]['MCEventNumber']) &
                         (data[keyLow]['ID'][i] == data[keyHigh]['ID']))[0]  # [0]
        if index.size == 0:
            data[keyLow]['DNNPredTrueClassNoInd'][i] = -2.0
        elif index.size > 1:
            print i, index, \
                data[keyLow]['DNNTrueClass'][i], data[keyLow]['DNNPredTrueClass'][i], \
                data[keyHigh]['DNNTrueClass'][index[0]], data[keyHigh]['DNNPredTrueClass'][index[0]], \
                data[keyHigh]['DNNTrueClass'][index[1]], data[keyHigh]['DNNPredTrueClass'][index[1]]
        else:
            for key in data[keyLow].keys():
                if 'DNN' in key: continue #Dont check DNN variables
                if 'Fiducial' in key: continue #Skip because inconsistent. Probably used different FV definition at SLAC
                if np.any( data[keyLow][key][i] != data[keyHigh][key][index[0]] ):
                    print i, key, data[keyLow][key][i], data[keyHigh][key][index[0]]
                    raw_input('')
            data[keyLow]['DNNPredTrueClassNoInd'][i] = data[keyHigh]['DNNPredTrueClass'][index[0]]

        if i % 10000 == 0:
            print i

    write_dict_to_hdf5_file(data[keyLow], folderRUNS + 'combined.hdf5')
    return data['w/ Ind']


##################################################################################################

# TODO reduced DNN
# folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/190131-1707/0validation/Compare_025_Induction/'
# files = {}
# files['w/ Ind'] = '../mixed-mc-reduced-025-small/events_025_mixed-mc-reduced-small.hdf5'
# files['w/o Ind'] = '../mixed-mc-reducedNoInd-025-small/events_025_mixed-mc-reducedNoInd-small.hdf5'

# TODO raw DNN
folderRUNS = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/TrainingRuns/180906-1938/0validation/Compare_023_Induction/'
files = {}
files['w/ Ind'] = '../mixed-mc-reduced-023-small/events_023_mixed-mc-reduced-small.hdf5'
files['w/o Ind'] = '../mixed-mc-reducedNoInd-023-small/events_023_mixed-mc-reducedNoInd-small.hdf5'

print '===== start reading'
data = GetData(files, False)

print '===== start selecting'
maskFid = {}
maskSS = {}
maskMS = {}
maskBKG = {}
maskSIG = {}
maskROI = {}
for key in files.keys():
    maskFid[key] = (np.sum(data[key]['CCIsFiducial'], axis=1) == data[key]['CCNumberClusters']) & (np.sum(data[key]['CCIs3DCluster'], axis=1) == data[key]['CCNumberClusters'])
    for key2 in data[key].keys():
        data[key][key2] = data[key][key2][maskFid[key]]
    maskSS[key] = (data[key]['CCIsSS'] == 1)
    maskMS[key] = np.invert(maskSS[key])
    maskBKG[key] = (data[key]['DNNTrueClass'] == 0)
    maskSIG[key] = (data[key]['DNNTrueClass'] == 1)
    maskROI[key] = (np.sum(data[key]['CCCorrectedEnergy'], axis=1) > 2400.) & \
                   (np.sum(data[key]['CCCorrectedEnergy'], axis=1) < 2800.)


kwargs = {
    'range': (0, 1),
    'bins': 30,
    'density': False
}

# ===== ROC
k, vp, vt = map(list, zip(*[ (key, data[key]['DNNPredTrueClass'], data[key]['DNNTrueClass']) for key in files.keys() ]))
kss, vpss, vtss= map(list, zip(*[ (key+'-SS', data[key]['DNNPredTrueClass'][maskSS[key]], data[key]['DNNTrueClass'][maskSS[key]]) for key in files.keys() ]))
kms, vpms, vtms = map(list, zip(*[ (key+'-MS', data[key]['DNNPredTrueClass'][maskMS[key]], data[key]['DNNTrueClass'][maskMS[key]]) for key in files.keys() ]))
plot_ROC_curve(fOUT=folderRUNS + 'roc_curve_DNN.pdf',
               dataTrue=vt+vtss+vtms,
               dataPred=vp+vpss+vpms,
               label=k+kss+kms)


# ===== Full Energy
ksig, vpsig = map(list, zip(*[ (key+'-Sig', data[key]['DNNPredTrueClass'][maskSIG[key] & maskSS[key]]) for key in files.keys() ]))
kbkg, vpbkg = map(list, zip(*[ (key+'-Bkg', data[key]['DNNPredTrueClass'][maskBKG[key] & maskSS[key]]) for key in files.keys() ]))
plot_histogram_vs_threshold(fOUT=folderRUNS + 'histogram_vs_threshold-SS.pdf',
                            data=vpsig+vpbkg, label=ksig+kbkg)

make_shape_agreement_plot([vpsig[0], vpbkg[0]], [vpsig[1], vpbkg[1]], [ksig[0], kbkg[0]], [ksig[1], kbkg[1]],'signal-likeness-SS', **kwargs)

ksig, vpsig = map(list, zip(*[ (key+'-Sig', data[key]['DNNPredTrueClass'][maskSIG[key] & maskMS[key]]) for key in files.keys() ]))
kbkg, vpbkg = map(list, zip(*[ (key+'-Bkg', data[key]['DNNPredTrueClass'][maskBKG[key] & maskMS[key]]) for key in files.keys() ]))
plot_histogram_vs_threshold(fOUT=folderRUNS + 'histogram_vs_threshold-MS.pdf',
                            data=vpsig+vpbkg, label=ksig+kbkg)

make_shape_agreement_plot([vpsig[0], vpbkg[0]], [vpsig[1], vpbkg[1]], [ksig[0], kbkg[0]], [ksig[1], kbkg[1]],'signal-likeness-MS', **kwargs)


# ===== ROI
ksig, vpsig = map(list, zip(*[ (key+'-Sig', data[key]['DNNPredTrueClass'][maskROI[key] & maskSIG[key] & maskSS[key]]) for key in files.keys() ]))
kbkg, vpbkg = map(list, zip(*[ (key+'-Bkg', data[key]['DNNPredTrueClass'][maskROI[key] & maskBKG[key] & maskSS[key]]) for key in files.keys() ]))
plot_histogram_vs_threshold(fOUT=folderRUNS + 'histogram_vs_threshold-SS-ROI.pdf',
                            data=vpsig+vpbkg, label=ksig+kbkg)

make_shape_agreement_plot([vpsig[0], vpbkg[0]], [vpsig[1], vpbkg[1]], [ksig[0], kbkg[0]], [ksig[1], kbkg[1]],'signal-likeness-SS-ROI', **kwargs)

ksig, vpsig = map(list, zip(*[ (key+'-Sig', data[key]['DNNPredTrueClass'][maskROI[key] & maskSIG[key] & maskMS[key]]) for key in files.keys() ]))
kbkg, vpbkg = map(list, zip(*[ (key+'-Bkg', data[key]['DNNPredTrueClass'][maskROI[key] & maskBKG[key] & maskMS[key]]) for key in files.keys() ]))
plot_histogram_vs_threshold(fOUT=folderRUNS + 'histogram_vs_threshold-MS-ROI.pdf',
                            data=vpsig+vpbkg, label=ksig+kbkg)

make_shape_agreement_plot([vpsig[0], vpbkg[0]], [vpsig[1], vpbkg[1]], [ksig[0], kbkg[0]], [ksig[1], kbkg[1]],'signal-likeness-MS-ROI', **kwargs)

# ===== Combined
print '===== start reading'
data = GetData(files, True)

print '===== start selecting'
maskFid = (data['DNNPredTrueClassNoInd'] > -1.5) & \
          (np.sum(data['CCIsFiducial'], axis=1) == data['CCNumberClusters']) & \
          (np.sum(data['CCIs3DCluster'], axis=1) == data['CCNumberClusters'])
for key in data.keys():
    data[key] = data[key][maskFid]

maskSS = (data['CCIsSS'] == 1)
maskMS = np.invert(maskSS)

maskROI = (np.sum(data['CCPurityCorrectedEnergy'], axis=1) > 2400.) & \
          (np.sum(data['CCPurityCorrectedEnergy'], axis=1) < 2800.)

maskBKG = (data['DNNTrueClass'] == 0)
maskSIG = (data['DNNTrueClass'] == 1)


# ===== Full Energy
plot_scatter_hist2d(data['DNNPredTrueClass'][maskSIG & maskSS],
                    data['DNNPredTrueClassNoInd'][maskSIG & maskSS],
                    'w/ Induction', 'w/o Induction', 'Signal (SS)', folderRUNS + 'hist2d_SS_SIG.pdf')

plot_scatter_hist2d(data['DNNPredTrueClass'][maskBKG & maskSS],
                    data['DNNPredTrueClassNoInd'][maskBKG & maskSS],
                    'w/ Induction', 'w/o Induction', 'Background (SS)', folderRUNS + 'hist2d_SS_BKG.pdf')

plot_scatter_hist2d(data['DNNPredTrueClass'][maskSIG & maskMS],
                    data['DNNPredTrueClassNoInd'][maskSIG & maskMS],
                    'w/ Induction', 'w/o Induction', 'Signal (MS)', folderRUNS + 'hist2d_MS_SIG.pdf')

plot_scatter_hist2d(data['DNNPredTrueClass'][maskBKG & maskMS],
                    data['DNNPredTrueClassNoInd'][maskBKG & maskMS],
                    'w/ Induction', 'w/o Induction', 'Background (MS)', folderRUNS + 'hist2d_MS_BKG.pdf')



plot_residual_hist2d(data['DNNPredTrueClass'][maskSIG & maskSS],
                     data['DNNPredTrueClassNoInd'][maskSIG & maskSS],
                     'w/ Induction', 'w/o', 'Signal (SS)', folderRUNS + 'residual2d_SS_SIG.pdf')

plot_residual_hist2d(data['DNNPredTrueClass'][maskBKG & maskSS],
                     data['DNNPredTrueClassNoInd'][maskBKG & maskSS],
                     'w/ Induction', 'w/o', 'Background (SS)', folderRUNS + 'residual2d_SS_BKG.pdf')

plot_residual_hist2d(data['DNNPredTrueClass'][maskSIG & maskMS],
                     data['DNNPredTrueClassNoInd'][maskSIG & maskMS],
                     'w/ Induction', 'w/o', 'Signal (MS)', folderRUNS + 'residual2d_MS_SIG.pdf')

plot_residual_hist2d(data['DNNPredTrueClass'][maskBKG & maskMS],
                     data['DNNPredTrueClassNoInd'][maskBKG & maskMS],
                     'w/ Induction', 'w/o', 'Background (MS)', folderRUNS + 'residual2d_MS_BKG.pdf')