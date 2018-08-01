#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dummy """

import numpy as np
import matplotlib.pyplot as plt
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator/')
import utilities.generator as gen

def main():
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_Uni_MC_P2/'
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/electron_WFs_Uni_MC_P2/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Plots/'

    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]

    EventInfo = gen.read_EventInfo_from_files(files) #, 5000)

    ys = np.asarray([EventInfo['QValue'],
                     EventInfo['MCPosX'],
                     EventInfo['MCPosY'],
                     EventInfo['MCPosZ']])

    print 'TOTAL:\t\t', ys.shape[1]
    print 'gamma:\t\t', ys[:, EventInfo['ID'] == 0].shape[1]
    print 'bb0nE:\t\t', ys[:, EventInfo['ID'] == 1].shape[1]
    print 'electron:\t', ys[:, EventInfo['ID'] == 2].shape[1]

    # plot_input_correlations(ys[:, EventInfo['ID'] == 0], folderOUT + 'Correlation_matrix_gamma.png')
    # plot_input_correlations(ys[:, EventInfo['ID'] == 1], folderOUT + 'Correlation_matrix_bb0nE.png')
    plot_input_correlations(ys[:, EventInfo['ID'] == 2], folderOUT + 'Correlation_matrix_electron.png')

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


def plot_input_correlations(ys, fileOUT):

    from pandas.plotting import scatter_matrix
    from pandas import DataFrame

    ys = np.swapaxes(ys, 0, 1)

    ys_data = DataFrame(ys, columns=['Energy', 'X-Position', 'Y-Position', 'Z-Position'])

    sm = scatter_matrix(ys_data, figsize=(25, 25), alpha=0.7, hist_kwds={'bins': 60}) #, diagonal='kde')

    for s in sm.reshape(-1):
        s.xaxis.label.set_size(16)
        s.yaxis.label.set_size(16)
        plt.setp(s.yaxis.get_majorticklabels(), 'size', 16)
        plt.setp(s.xaxis.get_majorticklabels(), 'size', 16)

    # plt.show()
    # plt.draw()
    # raw_input('')

    plt.savefig(fileOUT, bbox_inches='tight')

    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
