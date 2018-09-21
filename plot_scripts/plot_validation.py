#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from sys import path
path.append('/home/hpc/capm/sn0515/bbDiscriminator')
from plot_scripts.plot_traininghistory import *
from math import atan2,degrees

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def on_epoch_end_plots(folderOUT, epoch, data):
    # for i in xrange(200):
    #     print data['Y_TRUE'][i,1], data['Y_PRED'][i,1], data['Y_TRUE'][i,2], data['Y_PRED'][i,2]
    plot_scatter(data['Y_TRUE'][:,0], data['Y_PRED'][:,0], 'True Energy [keV]', 'DNN Energy [keV]', folderOUT+'prediction_energy_'+str(epoch)+'.png')
    plot_scatter(data['Y_TRUE'][:,1], data['Y_PRED'][:,1], 'True X [mm]', 'DNN X [mm]', folderOUT + 'prediction_X_'+str(epoch)+'.png')
    plot_scatter(data['Y_TRUE'][:,2], data['Y_PRED'][:,2], 'True Y [mm]', 'DNN Y [mm]', folderOUT + 'prediction_Y_'+str(epoch)+'.png')
    plot_scatter(data['Y_TRUE'][:,3], data['Y_PRED'][:,3], 'True Time [mu sec]', 'DNN Time [mu sec]', folderOUT + 'prediction_time_'+str(epoch)+'.png')
    plot_scatter(fromTimeToZ(data['Y_TRUE'][:, 3]), fromTimeToZ(data['Y_PRED'][:, 3]), 'True Z [mm]', 'DNN Z [mm]',
                 folderOUT + 'prediction_Z_' + str(epoch) + '.png')
    plot_traininghistory(folderOUT)
    return


def validation_mc_plots(args, folderOUT, data):
    name_DNN = 'DNN'
    name_EXO = 'EXO-Recon'
    name_True = 'True'
    peakpos = 2614.5

    kwargs = {
        'range': (0, 1),
        'bins': 100,
        'density': True
    }

    hist_ee, bin_edges = np.histogram(data['DNNPredTrueClass'][data['DNNTrueClass'] == 1], **kwargs)
    hist_y, bin_edges = np.histogram(data['DNNPredTrueClass'][data['DNNTrueClass'] == 0], **kwargs)

    hist_ee_SS, bin_edges = np.histogram(
        data['DNNPredTrueClass'][(data['DNNTrueClass'] == 1) & (data['CCIsSS'] == 1)], **kwargs)
    hist_y_SS, bin_edges = np.histogram(
        data['DNNPredTrueClass'][(data['DNNTrueClass'] == 0) & (data['CCIsSS'] == 1)], **kwargs)

    hist_ee_MS, bin_edges = np.histogram(
        data['DNNPredTrueClass'][(data['DNNTrueClass'] == 1) & (data['CCIsSS'] == 0)], **kwargs)
    hist_y_MS, bin_edges = np.histogram(
        data['DNNPredTrueClass'][(data['DNNTrueClass'] == 0) & (data['CCIsSS'] == 0)], **kwargs)

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # norm_ms = (data['CCIsSS'] == 1).sum()/float(len(data['CCIsSS']))

    plt.clf()
    plt.step(bin_centres, hist_y, where='mid', color='firebrick', label='gamma')
    plt.step(bin_centres, hist_ee, where='mid', color='blue', label='2beta')
    # plt.fill_between(bin_centres, np.zeros(hist_y_SS.shape), hist_y_MS*(1-norm_ms), label='gamma (MS)', step='mid', color='firebrick', alpha=0.3)
    # plt.fill_between(bin_centres, np.zeros(hist_ee_SS.shape), hist_ee_MS*(1-norm_ms), label='2beta (MS)', step='mid', color='blue', alpha=0.3)
    plt.xlabel('threshold')
    plt.legend(loc='upper center')
    plt.title('SS+MS')
    plt.xlim([0, 1])
    plt.ylim(ymin=0)
    plt.savefig(args.folderOUT + 'histogram_vs_threshold.pdf', bbox_inches='tight')
    plt.close()

    print data.keys()

    plt.clf()
    plt.step(bin_centres, hist_y_SS, where='mid', color='firebrick', label='gamma')
    plt.step(bin_centres, hist_ee_SS, where='mid', color='blue', label='2beta')
    plt.xlabel('threshold')
    plt.legend(loc='best')
    plt.title('SS-only')
    plt.xlim([0, 1])
    plt.ylim(ymin=0)
    plt.savefig(args.folderOUT + 'histogram_SS_vs_threshold.pdf', bbox_inches='tight')
    plt.close()

    plt.clf()
    plt.step(bin_centres, hist_y_MS, where='mid', color='firebrick', label='gamma')
    plt.step(bin_centres, hist_ee_MS, where='mid', color='blue', label='2beta')
    plt.xlabel('threshold')
    plt.legend(loc='best')
    plt.title('MS-only')
    plt.xlim([0, 1])
    plt.ylim(ymin=0)
    plt.savefig(args.folderOUT + 'histogram_MS_vs_threshold.pdf', bbox_inches='tight')
    plt.close()

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
        f1_score, accuracy_score, classification_report, precision_recall_curve, roc_curve, roc_auc_score

    energies = np.linspace(1000, 3000, 5, endpoint=True)
    eval_dict = {'cm': [], 'as': [], 'ps': [], 'rs': [], 'fs': [], 'prc': [], 'roc': [], 'roc_auc': []}
    for i in range(len(energies)):
        if i == 0:
            mask = np.ones(data['QValue'].size, dtype=bool)
            print 'Validating energies: %.0f - %.0f' % (energies[0], energies[-1])
        else:
            mask = np.asarray((data['QValue'] >= energies[i - 1]) & (data['QValue'] < energies[i]))
            print 'Validating energies: %.0f - %.0f' % (energies[i - 1], energies[i]), '\tNumber of events:', (
            mask == True).sum()

        eval_dict['cm'].append(confusion_matrix(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
        print eval_dict['cm'][-1]
        tn, fp, fn, tp = eval_dict['cm'][-1].ravel()
        print 'true negative (y->y)\t', tn
        print 'true positive (ee->ee)\t', tp
        print 'false positive (y->ee)\t', fp
        print 'false negative (ee->y)\t', fn

        eval_dict['as'].append(accuracy_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
        eval_dict['ps'].append(precision_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
        eval_dict['rs'].append(recall_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))
        eval_dict['fs'].append(f1_score(data['DNNTrueClass'][mask], data['DNNPredClass'][mask]))

        print 'accuracy score\t', eval_dict['as'][-1]
        print 'recall score\t', eval_dict['rs'][-1]
        print 'precision score\t', eval_dict['ps'][-1]
        print 'f1 score\t', eval_dict['fs'][-1]

        # cr = classification_report(data['DNNTrueClass'][mask], data['DNNPredClass'][mask], target_names=['gamma', 'bb'])
        # print cr

        eval_dict['prc'].append(
            precision_recall_curve(data['DNNTrueClass'][mask], data['DNNPredTrueClass'][mask]))

        eval_dict['roc'].append(roc_curve(data['DNNTrueClass'][mask], data['DNNPredTrueClass'][mask]))
        eval_dict['roc_auc'].append(
            roc_auc_score(data['DNNTrueClass'][mask], data['DNNPredTrueClass'][mask]))
        print 'roc auc score\t', eval_dict['roc_auc'][-1]

        print '========================================================'

    plt.clf()
    for i in range(len(energies[:-1])):
        plt.plot(eval_dict['prc'][i + 1][2], eval_dict['prc'][i + 1][0][:-1], '--', color='C%d' % i)
        plt.plot(eval_dict['prc'][i + 1][2], eval_dict['prc'][i + 1][1][:-1], '-', color='C%d' % i,
                 label='%.1f-%.1f MeV' % (energies[i] / 1.e3, energies[i + 1] / 1.e3))
    plt.plot(eval_dict['prc'][0][2], eval_dict['prc'][0][0][:-1], 'k--', lw=2)
    plt.plot(eval_dict['prc'][0][2], eval_dict['prc'][0][1][:-1], 'k-', lw=2, label='total')
    plt.xlabel('threshold')
    plt.legend(loc='lower left')
    plt.title('- - - - precision   |   $^{\_\_\_\_}$ recall')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(args.folderOUT + 'precision_recall_vs_threshold.pdf', bbox_inches='tight')
    plt.close()

    plt.clf()
    for i in range(len(energies[:-1])):
        plt.plot(eval_dict['prc'][i + 1][1], eval_dict['prc'][i + 1][0],
                 label='%.1f-%.1f MeV' % (energies[i] / 1.e3, energies[i + 1] / 1.e3))
    plt.plot(eval_dict['prc'][0][1], eval_dict['prc'][0][0], 'k-', lw=2, label='total')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='best')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(args.folderOUT + 'precision_vs_recall.pdf', bbox_inches='tight')
    plt.close()

    plt.clf()
    for i in range(len(energies[:-1])):
        plt.plot(eval_dict['roc'][i + 1][0], eval_dict['roc'][i + 1][1],
                 label='%.1f-%.1f MeV' % (energies[i] / 1.e3, energies[i + 1] / 1.e3))
    plt.plot(eval_dict['roc'][0][0], eval_dict['roc'][0][1], 'k-', lw=2, label='total')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(args.folderOUT + 'roc_curve.pdf', bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
# scatter
def plot_scatter(E_x, E_y, name_x, name_y, fOUT):
    dE = E_x - E_y
    diag = np.arange(min(E_x),max(E_x))
    plt.scatter(E_x, E_y, label='%s\n$\mu=%.1f, \sigma=%.1f$'%('training set', np.mean(dE), np.std(dE)))
    plt.plot(diag, diag, 'k--')
    plt.legend(loc="best")
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    # plt.xlim(xmin=600, xmax=3300)
    # plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

# scatter 2D heatmap
def plot_scatter_hist2d(E_x, E_y, name_x, name_y, name_title, fOUT):
    hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[[0,3300],[0,3300]], bins=250, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = (3300.0) / (3300.0)
    # plt.plot((500, 3200), (500, 3200), 'k--')
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    #cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    #cbar.set_label('Probability')
    plt.title('%s' % (name_title))
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=500, xmax=3300)
    plt.ylim(ymin=500, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return

#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        # ang = degrees(atan2(dy,dx))
        # ang = degrees(dy/dx)
        ang = 60.

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

# ----------------------------------------------------------
# Final Plots
# ----------------------------------------------------------
def plot_learning_curve(folderOUT, data):
    plt.clf()
    plt.plot(data[1], data[4], label='Validation')
    plt.plot(data[1], data[2], label='Training')
    plt.xlabel('Training time [epoch]')
    plt.ylabel('Loss')
    plt.grid(True, which='both')
    plt.xlim(xmin=0)
    # plt.gca().set_yscale('log')
    plt.legend(loc="best")
    plt.savefig(folderOUT + 'loss.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.axhline(y=50, color='k')
    plt.plot(data[1], 100. * data[5], label='Validation')
    plt.plot(data[1], 100.*data[3], label='Training')
    plt.grid(True)
    plt.ylim(ymin=40.0, ymax=100.0)
    plt.xlim(xmin=0)
    plt.legend(loc="best")
    plt.xlabel('Training time [epoch]')
    plt.ylabel('Accuracy [%]')
    plt.savefig(folderOUT + 'accuracy.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

# ----------------------------------------------------------
# Math Functions
# ----------------------------------------------------------
def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off

def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)

def erf(x, mu, sigma, B):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)

def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma

def gaussErf(x, A, mu, sigma, B):
    return gauss_zero(x, mu=mu, sigma=sigma, A=A) + erf(x, B=B, mu=mu, sigma=sigma)

def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]

def round_down(num, divisor):
    return num - (num%divisor)

def parabola(x, par0, par1, par2):
    return par0 + par1 * ((x - par2) ** 2)