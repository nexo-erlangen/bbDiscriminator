#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dummy """

import numpy as np
# import matplotlib as mpl
# mpl.use('PDF')
import matplotlib.pyplot as plt
import os
from sys import path
path.append('/home/hpc/capm/mppi053h/UVWireRecon')
import utilities.generator as gen

def main():
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/GammaExp_WFs_Uni_MC_SS/'
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/EnergyCorrectionNewNew/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/Dummy/'

    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]
    # print files
    number = gen.getNumEvents(files)/len(files)
    generator = gen.generate_batches_from_files(files, number, class_type='energy_and_UV_position', f_size=None, yield_mc_info=2)

    for idx in xrange(len(files)):
        # if idx >= 5:
        #     break
        print idx, 'of', len(files)
        # wf_temp, ys_temp, event_info = generator.next()
        event_info = generator.next()
        # print event_info['MCTime'][0:1,0]
        # print event_info['CCCollectionTime'][0:1,0]
        # print event_info['G4Time'][0:1,0]
        # print ''
        ys_temp = np.asarray([event_info['MCEnergy'][:,0],
                         event_info['MCPosU'][:,0],
                         event_info['MCPosV'][:,0],
                         event_info['MCPosZ'][:,0],
                         event_info['MCTime'][:,0]])
        # z_position = event_info['MCPosZ'][:, 0].reshape((1, 1))
        # ys_temp = np.append(ys_temp, z_position, axis=1)

        if idx == 0:
            ys = ys_temp
        else:
            ys = np.concatenate((ys, ys_temp),axis=1)
        # print ys.shape


    plot_input_correlations(ys, folderOUT)

    # timeToZfit(ys, folderOUT)


    # plot_input_correlations_heat(ys, folderOUT)
    return

def plot_input_correlations_heat(ys, folderOUT):
    plt.ion()
    plt.clf()

    # make Figure
    fig = plt.figure()

    # set size of Figure
    fig.set_size_inches(w=12*0.8, h=12*0.8)

    axarr = {}
    w = 0.8/5.
    h = 0.8/5.

    # add Axes
    for x in range(5):
        axarr[x] = {}
        for y in range(5):
            if x==0 and y==0:
                axarr[x, y] = fig.add_axes([x * 0.8 / 5.+0.1, y * 0.8 / 5.+0.1, w, h])
            if x==0 and y>0:
                axarr[x, y] = fig.add_axes([x * 0.8 / 5.+0.1, y * 0.8 / 5.+0.1, w, h], sharey=axarr[0, 0])
                plt.setp(axarr[x, y].get_xticklabels(), visible=False)
            if y==0 and x>0:
                axarr[x, y] = fig.add_axes([x * 0.8 / 5.+0.1, y * 0.8 / 5.+0.1, w, h], sharex=axarr[0, 0])
                plt.setp(axarr[x, y].get_yticklabels(), visible=False)
            if x>0 and y>0:
                axarr[x,y] = fig.add_axes([x*0.8/5.+0.1,y*0.8/5.+0.1,w,h], sharex=axarr[x,0], sharey=axarr[0,y])
                plt.setp(axarr[x, y].get_xticklabels(), visible=False)
                plt.setp(axarr[x, y].get_yticklabels(), visible=False)

    axarr[0, 0].set_xlim([500, 3500])
    axarr[1, 0].set_xlim([-200, 200])
    axarr[2, 0].set_xlim([-200, 200])
    axarr[3, 0].set_xlim([-200, 200])
    axarr[4, 0].set_xlim([1000, 2000])
    # axarr[0,0].set_ylim([-limit_Res, limit_Res])

    for i in range(5):
        axarr[i,i].hist(ys[i], bins=200, histtype="step", color="k", normed=True)

    # # ax1.hexbin(prEXO - trEXO, prXv - trXv, gridsize=(10*7,3*8), mincnt = 1, alpha = 0.7, vmin=0, vmax=100) norm=mpl.colors.LogNorm()
    # plt1 = ax1.hexbin(prEXO - trEXO, prXv - trXv, gridsize=400, mincnt = 1, norm=mpl.colors.LogNorm(), cmap=plt.get_cmap('viridis'), linewidths=0.1)
    # # fig.colorbar(plt1, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    #
    #
    #
    # ys_data = DataFrame(ys, columns=['Energy', 'x-Position', 'y-Position', 'Time', 'z-Position'])
    #
    # sm = scatter_matrix(ys_data, figsize=(25, 25), alpha=0.15, hist_kwds={'bins': 20})     # diagonal='kde')
    #
    # for s in sm.reshape(-1):
    #     s.xaxis.label.set_size(40)
    #     s.yaxis.label.set_size(40)
    #     plt.setp(s.yaxis.get_majorticklabels(), 'size', 20)
    #     plt.setp(s.xaxis.get_majorticklabels(), 'size', 20)

    plt.show()
    plt.draw()
    raw_input('')

    plt.savefig(folderOUT + 'Correlation_matrix' + '.png')

    # return


def plot_input_correlations(ys, folderOUT):

    from pandas.plotting import scatter_matrix
    from pandas import DataFrame

    ys = np.swapaxes(ys, 0, 1)

    ys_data = DataFrame(ys, columns=['Energy', 'U-Position', 'V-Position', 'z-Position', 'Time'])

    sm = scatter_matrix(ys_data, figsize=(25, 25), alpha=0.02, hist_kwds={'bins': 50})     # diagonal='kde')

    for s in sm.reshape(-1):
        s.xaxis.label.set_size(16)
        s.yaxis.label.set_size(16)
        plt.setp(s.yaxis.get_majorticklabels(), 'size', 16)
        plt.setp(s.xaxis.get_majorticklabels(), 'size', 16)

    # plt.show()
    # plt.draw()
    # raw_input('')

    plt.savefig(folderOUT + 'Correlation_matrix' + '.png')

    return

def linearfunc(x, m, t):
    return m * x + t

def timeToZfit(ys, folderOUT):

    from scipy.optimize import curve_fit
    import numpy as np

    popt, pcov = curve_fit(linearfunc, ys.T[:, 4], np.abs(ys.T[:, 3]))
    print popt
    print np.sqrt(np.diag(pcov))

    a = np.linspace(1000, 1200, 5)

    plt.plot(ys.T[:, 4], np.abs(ys.T[:, 3]), label='data', marker='.')
    plt.plot(a, linearfunc(a, *popt), color='red')
    plt.show()
    plt.draw()

    plt.clf()
    plt.scatter(ys.T[:, 4], -1.71 * ys.T[:, 4] + 1949.89)
    plt.show()
    plt.draw()

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
