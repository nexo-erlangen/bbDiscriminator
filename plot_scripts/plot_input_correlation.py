#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" dummy """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator/')
import utilities.generator as gen

EXOPHASE = 2
position = 'reducedNoInd'
DataOrMC = 'MC'
source = 'mixed'

def main():
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/%s_WFs_%s_%s_P%s/'%(source, position, DataOrMC, str(EXOPHASE))
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Plots/'

    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]

    print 'start loading'
    EventInfo = gen.read_EventInfo_from_files(files) #, 500000)
    print 'end loading'

#    EventInfo['MCEnergy'] = np.sum(EventInfo['CCPurityCorrectedEnergy'], axis=1)
#    EventInfo['MCPosX'] = np.ma.masked_equal(EventInfo['CCPosX'], 0.0).mean(axis=1)
#    EventInfo['MCPosY'] = np.ma.masked_equal(EventInfo['CCPosY'], 0.0).mean(axis=1)
#    EventInfo['MCPosZ'] = np.ma.masked_equal(EventInfo['CCPosZ'], 0.0).mean(axis=1)

    # mask = (EventInfo['MCEnergy'] > 2300.) & (EventInfo['MCEnergy'] < 2600.)
    mask = (EventInfo['MCEnergy'] >= 1000.) & (EventInfo['MCEnergy'] <= 3000.)
    #mask = (np.sum(EventInfo['CCPurityCorrectedEnergy'], axis=1) >= 2500.) &\
    #        (np.sum(EventInfo['CCPurityCorrectedEnergy'], axis=1) <= 2800.)

    ys = np.asarray([EventInfo['MCEnergy'], # TODO use QValue ?
                     EventInfo['MCPosX'],
                     EventInfo['MCPosY'],
                     EventInfo['MCPosZ']])

    print 'TOTAL:\t\t', ys.shape[1]
    print 'gamma:\t\t', ys[:, (EventInfo['ID'] == 0) & mask].shape[1]
    print 'bb0n:\t\t', ys[:, (EventInfo['ID'] == 1) & mask].shape[1]
    print 'electron:\t', ys[:, (EventInfo['ID'] == 2) & mask].shape[1]

    if source != 'mixed':
        plot_input_correlations_heat(ys[:, mask], folderOUT + 'Correlation_matrix_%s-P%s-%s-%s-heat.png'%(source, str(EXOPHASE), position, DataOrMC))
    else:
        plot_input_correlations_heat(ys[:, (EventInfo['ID'] == 0) & mask],
                                     folderOUT + 'Correlation_matrix_%s-P%s-%s-%s-heat.png' % (
                                     source + '-gamma', str(EXOPHASE), position, DataOrMC))
        plot_input_correlations_heat(ys[:, (EventInfo['ID'] == 1) & mask],
                                     folderOUT + 'Correlation_matrix_%s-P%s-%s-%s-heat.png' % (
                                         source + '-bb0nE', str(EXOPHASE), position, DataOrMC))

    return

def plot_input_correlations_heat(ys, fileOUT):
    PosMin = -180. #-180.
    PosMax = 180. #180.
    EneMin = 1000. #1000.
    EneMax = 3000.

    labelDict={}
    labelDict[0] = 'Energy [keV]'
    labelDict[1] = 'X [mm]'
    labelDict[2] = 'Y [mm]'
    labelDict[3] = 'Z [mm]'



    plt.clf()

    # make Figure
    fig = plt.figure()

    # set size of Figure
    fig.set_size_inches(w=12*0.8, h=12*0.8)

    axarr = {}
    w = 0.8/4.
    h = 0.8/4.

    # axarr_dummy = fig.add_axes([0 * 0.8 / 4. + 0.1, 0 * 0.8 / 4. + 0.1, w, h])
    # axarr_dummy.set_ylim([PosMin, PosMax])
    # axarr_dummy.set_ylim([EneMin, EneMax])
    # axarr_dummy.set_xticks([])
    # axarr_dummy.set_yticks([])
    # plt.setp(axarr_dummy.get_yticklabels(), visible=False)
    # plt.setp(axarr_dummy.get_xticklabels(), visible=False)

    # add Axes
    for x in range(4):
        axarr[x] = {}
        for y in range(4):
            # if x==0 and y==0:
            axarr[x, y] = fig.add_axes([x * 0.8 / 4.+0.1, y * 0.8 / 4.+0.1, w, h])

            if x == y and x == 0:
                axarr[x, y].set_xlim([EneMin, EneMax])
                plt.setp(axarr[x, y].get_yticklabels(), visible=False)
            elif x == y and x > 0:
                axarr[x, y].set_xlim([PosMin, PosMax])
                plt.setp(axarr[x, y].get_xticklabels(), visible=False)
                plt.setp(axarr[x, y].get_yticklabels(), visible=False)
            elif x > 0 and y > 0:
                plt.setp(axarr[x, y].get_xticklabels(), visible=False)
                plt.setp(axarr[x, y].get_yticklabels(), visible=False)
                axarr[x, y].set_xlim([PosMin, PosMax])
                axarr[x, y].set_ylim([PosMin, PosMax])
            elif x == 0 and y > 0:
                plt.setp(axarr[x, y].get_xticklabels(), visible=False)
                axarr[x, y].set_xlim([EneMin, EneMax])
                axarr[x, y].set_ylim([PosMin, PosMax])
            elif y == 0 and x > 0:
                plt.setp(axarr[x, y].get_yticklabels(), visible=False)
                axarr[x, y].set_xlim([PosMin, PosMax])
                axarr[x, y].set_ylim([EneMin, EneMax])
            else:
                print x , y

    for i in range(4):
        axarr[i,i].hist(ys[i], bins=40, histtype="step", color="k", normed=True)
        axarr[i, 0].set_xlabel(labelDict[i])
        axarr[0, i].set_ylabel(labelDict[i])

    for x in range(4):
        for y in range(4):
            plt.setp(axarr[x, y].get_xticklabels()[0], visible=False)
            plt.setp(axarr[x, y].get_xticklabels()[-1], visible=False)
            plt.setp(axarr[x, y].get_yticklabels()[0], visible=False)
            plt.setp(axarr[x, y].get_yticklabels()[-1], visible=False)
            if x==y: continue
            axarr[x,y].hexbin(ys[x], ys[y], gridsize=40, mincnt = 1, norm=colors.Normalize(), cmap=plt.get_cmap('viridis'), linewidths=0.1)

    plt.savefig(fileOUT, bbox_inches='tight')

    return


def plot_input_correlations(ys, fileOUT):

    from pandas.plotting import scatter_matrix
    from pandas import DataFrame

    ys = np.swapaxes(ys, 0, 1)

    ys_data = DataFrame(ys, columns=['Energy', 'X-Position', 'Y-Position', 'Z-Position'])

    sm = scatter_matrix(ys_data, figsize=(25, 25), alpha=0.3, hist_kwds={'bins': 60}) #, diagonal='kde')

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
