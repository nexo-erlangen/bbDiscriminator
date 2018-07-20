#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
from scipy.optimize import curve_fit

def exponential(x, par0, par1):
    return par0 * np.exp(x/par1)

def linear(x, par0, par1):
    return x/par1 + np.log(par0)

def exponentiallin(x, par0, par1, par2):
    return par0 * np.exp(x/par1) + par2

def exponentials(x, par0, par1, par2):
    return par0 * ( np.exp(x/par1) + np.exp(x/par2) )

def exponentialpot2(x, par0, par1, par3):
    return par0 * (np.power(x-par3,2.0)) * np.exp((x-par3)/par1)

def exponentialpot3(x, par0, par1, par3):
    return par0 * (np.power(x-par3,3.0)) * np.exp((x-par3)/par1)

folderOLD = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/EnergyCorrectionOLD/'
folder = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/EnergyCorrection/'
folderNew = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/EnergyCorrectionNew/'
filesOLD = [f for f in listdir(folderOLD) if isfile(join(folderOLD, f)) and '.hdf5' in f]
files = [f for f in listdir(folder) if isfile(join(folder, f)) and '.hdf5' in f]
filesNew = [f for f in listdir(folderNew) if isfile(join(folderNew, f)) and '.hdf5' in f]

for index, filename in enumerate(filesNew):
    fIN = h5py.File(folderNew + str(filename), "r")
    if index == 0:
        dataNew = np.asarray(fIN['MCEnergy'][:,0])
    else:
        dataNew = np.concatenate((dataNew, np.asarray(fIN['MCEnergy'][:,0])))
    fIN.close()
print dataNew.shape

for index, filename in enumerate(files):
    fIN = h5py.File(folder + str(filename), "r")
    if index == 0:
        data = np.asarray(fIN['MCEnergy'][:,0])
    else:
        data = np.concatenate((data, np.asarray(fIN['MCEnergy'][:,0])))
    fIN.close()
print data.shape

for index, filename in enumerate(filesOLD):
    fIN = h5py.File(folderOLD + str(filename), "r")
    if index == 0:
        dataOLD = np.asarray(fIN['MCEnergy'][:,0])
    else:
        dataOLD = np.concatenate((dataOLD, np.asarray(fIN['MCEnergy'][:,0])))
    fIN.close()
print dataOLD.shape

plt.ion()
hist, bin_edges = np.histogram(data/1.e3, bins=25, range=(0.55, 3.5), density=True)
histOLD, bin_edges = np.histogram(dataOLD/1.e3, bins=25, range=(0.55, 3.5), density=True)
histNew, bin_edges = np.histogram(dataNew/1.e3, bins=25, range=(0.55, 3.5), density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

coeff = [2.e0, -0.9]
for i in range(5):
    try:
        coeff, var_matrix = curve_fit(exponential, bin_centres[:-1], histOLD[:-1], p0=coeff)
        coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
    except:
        print 'fit did not work'

print coeff
print coeff_err
print abs(coeff_err/coeff)*100.
print '======================================='

# coefflin = [-8.e-4, -2.2]
# for i in range(5):
#     try:
#         coefflin, var_matrix = curve_fit(linear, bin_centres[:-1], np.log10(histOLD[:-1]), p0=coefflin)
#         coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
#     except:
#         print 'fit did not work'
#
# print coefflin
# print coeff_err
# print abs(coeff_err/coefflin)*100.
# print '======================================='
#
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

smooth = savgol_filter(histOLD[5:-5], 3, 1)
f2 = interp1d(bin_centres[5:-5], smooth, kind='linear', bounds_error=False, fill_value='extrapolate')
# f3 = Akima1DInterpolator(bin_centres, hist)

fine_bins, bin_size = np.linspace(0.5,3.5,1024,endpoint=True, retstep=True)


# coefflin = [5.e0, -0.5]
# for i in range(5):
#     try:
#         coefflin, var_matrix = curve_fit(linear, bin_centres[:-1], np.log(histOLD[:-1]), p0=coefflin)
#         coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
#     except:
#         print 'fit did not work'
#
# print coefflin
# print coeff_err
# print abs(coeff_err / coefflin) * 100.
# print '======================================='


# fine_bins = np.linspace(bin_centres[0], bin_centres[-1],1024)

plt.clf()
# plt.step(bin_centres, histOLD, where='mid', label='old')
plt.step(bin_centres, hist, where='mid', label='new')
plt.step(bin_centres, histNew, where='mid', label='newNew')

# plt.plot(bin_centres[:-1], exponential(bin_centres[:-1], *coeff), label='Fit')
# plt.plot(bin_centres[:-1], np.exp(linear(bin_centres[:-1], *coefflin)), label='Fit', color='blue')
# plt.scatter(fine_bins, linear(fine_bins, *coefflin), label='newFit', marker='.')
# plt.step(bin_centres, np.log10(histOLD), where='mid', label='old')
# plt.plot(bin_centres, np.log10(exponential(bin_centres, *coeff)), label='exp')
# plt.plot(bin_centres, linear(bin_centres, *coefflin), label='lin')

# plt.scatter(fine_bins, f2(fine_bins), label='inter cubic', marker='.')
# plt.scatter(fine_bins, exponential(fine_bins, *coefflin), label='newFit', marker='.', color='blue')
# plt.scatter(bin_centres[5:-5], smooth, label='savgol', color='red', marker='x')
# plt.scatter(fine_bins, f2(fine_bins)*linear(fine_bins, *coefflin)/np.sum(f2(fine_bins)*linear(fine_bins, *coefflin)), label='combination', marker='.', color='red')
# plt.scatter(fine_bins, f2(fine_bins)+linear(fine_bins, *coefflin)/np.sum(f2(fine_bins)+linear(fine_bins, *coefflin)), label='combination', marker='.', color='blue')


# plt.gca().set_yscale('log')
plt.legend(loc="best")
plt.xlim(xmin=0.45, xmax=3.55)
# plt.ylim(ymin=1.e-6, ymax=1.e-2)
plt.show()
plt.draw()
raw_input('')

weights = 1./f2(fine_bins)
weights /= np.sum(weights)

print weights
print np.sum(weights)

print '============================'
for i in xrange(len(fine_bins)):
    print '/gps/hist/point', fine_bins[i], weights[i]*1.e5
print '============================'
print len(fine_bins)

plt.clf()
plt.plot(fine_bins,weights)
plt.scatter(fine_bins, weights, marker='.', color='blue')
plt.gca().set_yscale('log')
plt.xlim(xmin=0.45, xmax=3.55)
plt.ylim(ymin=1.e-6, ymax=1.e-1)
plt.show()
plt.draw()
raw_input('')


# plt.clf()
# plt.plot(fine_bins_centers, weights*f2(fine_bins_centers), label='inter cubic')
# plt.gca().set_yscale('log')
# plt.legend(loc="best")
# plt.xlim(xmin=450, xmax=3550)
# # plt.ylim(ymin=1.e-6, ymax=1.e-2)
# plt.show()
# plt.draw()
# raw_input('')







