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

def main():
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/shuffle_bb0nE_gamma_WFs_MC_P2-short/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Waveforms/'
    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f)) and '.hdf5' in f]
    number = 3000
    generator = gen.generate_batches_from_files(files, 1, class_type=None, f_size=None, yield_mc_info=1)
    for idx in xrange(number):
        print 'plot waveform \t', idx
        wf, _, eventInfo = generator.next()
        if eventInfo['LXeEnergy'] < 2800 or ( abs(eventInfo['MCPosZ']) > 15 and abs(eventInfo['MCPosZ']) < 175 ): continue
        # if eventInfo['LXeEnergy'] < 2800 or abs(eventInfo['MCPosZ']) < 175: continue
        if eventInfo['ID'] == 0: particleID = 'Photon'
        elif eventInfo['ID'] == 1: particleID = 'doubleBeta'
        plot_waveforms(np.asarray(wf), idx, particleID, eventInfo['LXeEnergy'], folderOUT)
    return

def plot_waveforms(wf, idx, partID, energy, folderOUT):
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

    fig.savefig(folderOUT + str(idx) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()