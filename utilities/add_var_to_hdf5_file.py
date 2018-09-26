#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility function to crop images to ROI."""

import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import h5py

batchsize = 1
fletcher32 = True
shuffle = False
compression = ('gzip', 4)

def main():
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_Uni_MC_P2/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/mixed_WFs_Uni_MC_P2-withBDT/'

    folderBDT = '/home/vault/capm/sn0515/PhD/DeepLearning/bbDiscriminator/Data/bdt_combined_hdf5/'
    file_bdt_signal = folderBDT + 'bb0n_withBDt.hdf5'
    file_bdt_bkg = folderBDT + 'gamma_withBDt.hdf5'

    print
    print 'Input Folder:\t', folderIN
    print 'Output Folder:\t', folderOUT
    print

    files = [f for f in listdir(folderIN) if isfile(join(folderIN, f)) and '.hdf5' in f]

    print 'Number of Files:\t', len(files)
    print

    print 'BDT Signal File:\t', file_bdt_signal
    print 'BDT Bkg File:\t\t', file_bdt_bkg
    print

    data_bdt_signal = {}
    f = h5py.File(str(file_bdt_signal), "r")
    for key in f.keys():
        data_bdt_signal[key] = np.asarray(f[key])
    f.close()

    data_bdt_bkg = {}
    f = h5py.File(str(file_bdt_bkg), "r")
    for key in f.keys():
        data_bdt_bkg[key] = np.asarray(f[key])
    f.close()

    print 'BDT Signal Number Events:\t', data_bdt_signal.values()[0].shape[0]
    print 'BDT Bkg Number Events:\t\t', data_bdt_bkg.values()[0].shape[0]
    print


    for file in files:
        print 'adjusting file:\t\t', file
        add_var_to_file(folderIN+file, folderOUT+file, data_sig=data_bdt_signal, data_bkg=data_bdt_bkg)


def add_var_to_file(fileIN, fileOUT, data_sig, data_bkg):
    fIN = h5py.File(fileIN, "r")
    fOUT = h5py.File(fileOUT, "w")

    fIN_RunNum = np.asarray(fIN.get('MCRunNumber'))
    fIN_EventNum = np.asarray(fIN.get('MCEventNumber'))
    fIN_ID = np.asarray(fIN.get('ID'))
    fIN_NumEvents = len(fIN_EventNum)

    keyID = {}
    keyID['numColl'] = 'CCNumCollWires'
    keyID['risetime'] = 'CCRisetime'
    keyID['maxV'] = 'CCMaxVFraction'
    keyID['stand'] = 'CCStandoff'
    keyID['bdt_ss'] = 'BDT-SS'
    keyID['bdt_ss_withV'] = 'BDT-SS-V'
    keyID['bdt_ss_noStand'] = 'BDT-SS-NoStandoff'
    keyID['bdt_all'] = 'BDT-SSMS'

    fOUT_new = {}
    for key in data_sig.keys():
        if key in ['runNum', 'eventNum']: continue
        elif key not in keyID.keys(): raise ValueError('strange key: %s'%key)
        fOUT_new[keyID[key]] = np.zeros((fIN_NumEvents,), dtype=np.float32)

    data = {}
    data[0] = data_bkg
    data[1] = data_sig

    for i in xrange(len(fIN_EventNum)):
        id = fIN_ID[i]
        index = np.where(np.logical_and(fIN_RunNum[i] == data[id]['runNum'],
                                        fIN_EventNum[i] == data[id]['eventNum']))[0][0]

        for key in data[id].keys():
            if key in ['runNum', 'eventNum']: continue
            fOUT_new[keyID[key]][i] = data[id][key][index]

    wfs = np.asarray(fIN.get('wfs'))
    chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None

    fOUT.create_dataset('wfs', data=wfs, dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in fIN.keys():
        if key in ['wfs']: continue
        temp = np.asarray(fIN.get(key))
        fOUT.create_dataset(key, data=temp, dtype=np.float32)

    for key in fOUT_new.keys():
        fOUT.create_dataset(key, data=fOUT_new[key], dtype=np.float32)

    fIN.close()
    fOUT.close()



if __name__ == '__main__':
    main()