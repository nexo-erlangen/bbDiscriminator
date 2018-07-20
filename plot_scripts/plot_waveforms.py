#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import os
from sys import path
path.append('/home/hpc/capm/sn0515/UVWireRecon')

from utilities import generator as gen

def main():
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/UniformGamma_ExpWFs_MC_SS/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Waveforms/'
    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]
    number = 100
    generator = gen.generate_batches_from_files(files, 1, class_type='energy_and_position', f_size=None, yield_mc_info=False)
    for idx in xrange(number):
        print 'plot waveform \t', idx
        wf, _ = generator.next()
        plot_waveforms(np.asarray(wf), idx, folderOUT)
    return

def plot_waveforms(wf, idx, folderOUT):
    time = range(0, 2048)

    # xs_i = np.swapaxes(xs_i, 0, 1)
    wf = np.swapaxes(wf, 1, 2)
    wf = np.swapaxes(wf, 2, 3)

    plt.clf()
    # make Figure
    fig, axarr = plt.subplots(2, 2)

    # set size of Figure
    fig.set_size_inches(w=20., h=8.)

    for i in xrange(wf.shape[0]):
        if i == 0 : x = 1; y = 0
        elif i == 1: x = 0; y = 0
        elif i == 2: x = 1; y = 1
        elif i == 3: x = 0; y = 1
        axarr[x, y].set_xlim([0.0, 2048])
        axarr[x, y].set_ylim([-40, 780])
        plt.setp(axarr[x, y].get_yticklabels(), visible=False)
        for j in xrange(wf.shape[2]):
            axarr[x, y].plot(time, wf[i, : , j, 0, 0] + 20. * j, color='k')

    axarr[1, 0].set_ylabel(r'Amplitude + offset [a.u.]')
    axarr[0, 0].set_ylabel(r'Amplitude + offset [a.u.]')
    axarr[1, 1].set_xlabel(r'Time [$\mu$s]')
    axarr[1, 0].set_xlabel(r'Time [$\mu$s]')

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    fig.savefig(folderOUT + str(idx) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()