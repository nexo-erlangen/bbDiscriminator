#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator/')

from utilities import generator as gen

TimeOffset = 1000

# mode = 'single'
mode = 'sum'

def main():
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_AllVessel_MC_P2/'
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_reduced_MC_P2/'
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/Th228_WFs_S5_Data_P2-old/'
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/Th228_WFs_S5_Data_P2/'
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/Th228_WFs_S5_MC_P2/'
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/Co60_WFs_S5_Data_P2/'
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/LB_WFs_AllVessel_Data_P2/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Waveforms/'
    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f)) and '.hdf5' in f]
    number = 156000
    generator = gen.generate_batches_from_files(files, 1, wires='U', class_type=None, f_size=None, yield_mc_info=1)
    wf_sum = [None]*3 
    for idx in xrange(number):
        wf, _, eventInfo = generator.next()
        wf = np.asarray(wf)
        eventInfo['ID'] = int(eventInfo['ID'])
        # if eventInfo['LXeEnergy'] < 2800 or ( abs(eventInfo['MCPosZ']) > 15 and abs(eventInfo['MCPosZ']) < 175 ): continue
        # if eventInfo['LXeEnergy'] < 2800 or abs(eventInfo['MCPosZ']) < 175: continue
        if eventInfo['ID'] == 0: particleID = 'Photon'
        elif eventInfo['ID'] == 1: particleID = 'doubleBeta'
        elif eventInfo['ID'] == 2: particleID = 'Electron'

        if mode == 'single':
            print 'plot waveform \t', idx
            if wf.shape[0] == 2: #plot_waveforms_U(np.asarray(wf), idx, particleID, eventInfo['LXeEnergy'], folderOUT)
                plot_waveforms_U(np.asarray(wf), idx, particleID, 0.0, folderOUT)
            elif wf.shape[0] == 4: plot_waveforms_UV(np.asarray(wf), idx, particleID, eventInfo['MCEnergy'], folderOUT)
            else: ValueError('strange waveform shape: %s'%(wf.shape))
        elif mode == 'sum':
            if idx%1000==0: print idx
            if wf_sum[eventInfo['ID']] is None:
                print 'initializing wfs:', eventInfo['ID']
                wf_sum[eventInfo['ID']] = wf
            else:
                wf_sum[eventInfo['ID']] += wf
            # print wf.shape
    if mode == 'sum':
        for i in range(len(wf_sum)):
            if wf_sum[i] is None: continue
            plot_waveforms_heat(np.asarray(wf_sum[i]), folderOUT + 'heat-Co60-Data-S5-%s.png'%(str(i)))
    return

def plot_waveforms_UV(wf, idx, partID, energy, folderOUT):
    # xs_i = np.swapaxes(xs_i, 0, 1)
    wf = np.swapaxes(wf, 1, 2)
    wf = np.swapaxes(wf, 2, 3)
    # print wf.shape

    time = np.arange(TimeOffset, TimeOffset+wf.shape[1])
    # print time.shape

    plt.clf()
    # make Figure
    fig, axarr = plt.subplots(2, 2)

    # set size of Figure
    fig.set_size_inches(w=20., h=8.)
    fig.suptitle("ID: %s     Energy: %d keV" % (partID, energy), fontsize=18)

    for i in xrange(wf.shape[0]):
        if i == 0 : x = 1; y = 0; title='U-Wires TPC 1'
        elif i == 1: x = 0; y = 0; title='V-Wires TPC 1'
        elif i == 2: x = 1; y = 1; title='U-Wires TPC 2'
        elif i == 3: x = 0; y = 1; title='V-Wires TPC 2'
        axarr[x, y].set_xlim([TimeOffset, TimeOffset+wf.shape[1]])
        axarr[x, y].set_ylim([-40, 780])
        axarr[x, y].set_title(title, fontsize=16)
        plt.setp(axarr[x, y].get_yticklabels(), visible=False, fontsize=16)
        for j in xrange(wf.shape[2]):
            axarr[x, y].plot(time, wf[i, : , j, 0, 0] + 20. * j, color='k')
        axarr[x, y].axvline(x=1000, color='k', lw=2)
        axarr[x, y].axvline(x=1350, color='k', lw=2)

    axarr[1, 0].set_ylabel(r'Amplitude + offset [a.u.]', fontsize=16)
    axarr[0, 0].set_ylabel(r'Amplitude + offset [a.u.]', fontsize=16)
    axarr[1, 1].set_xlabel(r'Time [$\mu$s]', fontsize=16)
    axarr[1, 0].set_xlabel(r'Time [$\mu$s]', fontsize=16)

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fig.savefig(folderOUT + partID + '_UV' + str(idx) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

def plot_waveforms_U(wf, idx, partID, energy, folderOUT):
    # xs_i = np.swapaxes(xs_i, 0, 1)
    wf = np.swapaxes(wf, 1, 2)
    wf = np.swapaxes(wf, 2, 3)

    time = np.arange(TimeOffset, TimeOffset+wf.shape[1])

    plt.clf()
    # make Figure
    fig, axarr = plt.subplots(1, 2)

    # set size of Figure
    fig.set_size_inches(w=20., h=5.)
    fig.suptitle("ID: %s     Energy: %d keV" % (partID, energy), fontsize=18)

    for i in xrange(wf.shape[0]):
        if i == 0 : title='U-Wires TPC 1'
        elif i == 1: title='U-Wires TPC 2'
        axarr[i].set_xlim([TimeOffset, TimeOffset+wf.shape[1]])
        axarr[i].set_ylim([-40, 780])
        axarr[i].set_title(title, fontsize=16)
        plt.setp(axarr[i].get_yticklabels(), visible=False, fontsize=16)
        for j in xrange(wf.shape[2]):
            axarr[i].plot(time, wf[i, : , j, 0, 0] + 20. * j, color='k')
        axarr[i].axvline(x=1000, color='k', lw=2)
        axarr[i].axvline(x=1350, color='k', lw=2)

    axarr[0].set_ylabel(r'Amplitude + offset [a.u.]', fontsize=16)
    # axarr[1].set_ylabel(r'Amplitude + offset [a.u.]', fontsize=16)
    axarr[0].set_xlabel(r'Time [$\mu$s]', fontsize=16)
    axarr[1].set_xlabel(r'Time [$\mu$s]', fontsize=16)

    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fig.savefig(folderOUT + partID + '_U' + str(idx) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

def plot_waveforms_heat(data, fOUT):
    from matplotlib import gridspec, colors

    range_x = [0, 350]
    range_y = [0, 38]
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]

    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 2, 3)
    data = np.squeeze(data)
    data = np.squeeze(data)

    aspect = "auto"
    data /= np.max(data)

    plt.clf()
    f, axarr = plt.subplots(1, 2)
    f.set_size_inches(w=20., h=5.)
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharey=ax1)

    h1 = ax1.imshow(data[0].T, extent=extent, interpolation='nearest', vmin=-1, vmax=1, cmap=plt.get_cmap('RdBu_r'), origin='lower', aspect=aspect) #, norm=colors.Normalize(vmax=1))
    h2 = ax2.imshow(data[1].T, extent=extent, interpolation='nearest', vmin=-1, vmax=1, cmap=plt.get_cmap('RdBu_r'), origin='lower', aspect=aspect) #, norm=colors.Normalize(vmax=1))
    f.colorbar(h2, ax=ax2, shrink=0.6)

    ax1.set_xlim(range_x)
    ax2.set_xlim(range_x)
    ax1.set_ylim(range_y)

    ax1.set_ylabel(r'Amplitude + offset [a.u.]', fontsize=16)
    ax1.set_xlabel(r'arbitrary time [$\mu$s]', fontsize=16)
    ax2.set_xlabel(r'arbitrary time [$\mu$s]', fontsize=16)

    plt.setp(ax2.get_yticklabels(), visible=False)
    # plt.subplots_adjust(hspace=.0)
    f.savefig(fOUT, bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
