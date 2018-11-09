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
    folderIN = os.path.join(sys.argv[1], '')
    folderOUT = os.path.join(sys.argv[2], '')

    print
    print 'Input Folder:\t', folderIN
    print 'Output Folder:\t', folderOUT
    print

    files = [f for f in listdir(folderIN) if isfile(join(folderIN, f)) and '.hdf5' in f]

    print 'Number of Files:\t', len(files)
    print

    offset = 0 #512
    start = 1000 # 1000
    length = 350 #350

    # for i, file in enumerate(files):
    #     print 'cropping file:\t\t', file, 'to', str(i+18), '.hdf5'
    #     crop_image(folderIN+file, folderOUT+str(i+18)+'.hdf5', start=start-offset, length=length)
    #     # break

    for file in files:
        print 'cropping file:\t\t', file
        # if file in ['0.hdf5', '1.hdf5', '2.hdf5']: continue
        # if file not in ['7126_1.hdf5']: continue
        # if isfile(folderOUT+file): continue
        crop_image(folderIN+file, folderOUT+file, start=start-offset, length=length)
        # break


def crop_image(fileIN, fileOUT, start, length):
    fIN = h5py.File(fileIN, "r")
    if 'wfs' not in fIN.keys(): raise AttributeError("wfs dataset not in file")
    fOUT = h5py.File(fileOUT, "w")

    MC = 1024.8234
    data = np.asarray(fIN['APDTime'])
    print data
    diff = MC-data
    print start
    print diff
    start_i = np.rint(start-(MC-data)).astype(int)
    print start_i
    print start_i.shape
    print

    wfs = np.asarray(fIN.get('wfs'))
    print wfs.shape
    # print 'changing length from:\t', wfs.shape[3], ' to ', length

    wfs_temp = np.zeros((wfs.shape[0], wfs.shape[1], length), dtype=np.float32)
    for i in xrange(wfs.shape[0]):
        try:
            # wfs_temp[i] = wfs[i, :, :, start_i[i]:start_i[i] + length]
            wfs_temp[i] = wfs[i, :, start_i[i]:start_i[i] + length]
        except:
            print i, start_i[i], length, MC, data[i]
            # wfs_temp[i] = wfs[i, :, :, start:start + length]
            wfs_temp[i] = wfs[i, :, start:start + length]
        if i%1000==0:
            print i, wfs_temp[i].shape
    print wfs.shape
    print wfs_temp.shape
    wfs = wfs_temp
    print wfs.shape

    # wfs = wfs[:, :, :, start:start + length]

    chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None
    print 'chunk size:\t', chunks_wfs

    fOUT.create_dataset("wfs", data=wfs, dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in fIN.keys():
        if key in ['wfs']: continue
        temp = np.asarray(fIN.get(key))
        # elif key in ['LXeEnergy', 'CCIs3DCluster', 'ID', 'APDEnergy', 'QValue']: continue
        # if key in ['MCEventNumber']:
        #     key = 'EventNumber'
        # elif key in ['MCRunNumber']:
        #     key = 'RunNumber'
        fOUT.create_dataset(key, data=temp, dtype=np.float32)

        # fIN_NumEvents = len(np.asarray(fIN.get('MCEventNumber')))
    # for key in ['MCEnergy', 'MCPosU', 'MCPosV', 'MCPosX', 'MCPosY', 'MCPosZ']:
    #     if key in fIN.keys(): continue
    #     temp = np.zeros((fIN_NumEvents,), dtype=np.float32)
    #     fOUT.create_dataset(key, data=temp, dtype=np.float32)
    #
    # # IsMC = np.zeros((fIN_NumEvents,), dtype=np.float32)
    # IsMC = np.ones((fIN_NumEvents,), dtype=np.float32)
    # fOUT.create_dataset('IsMC', data=IsMC, dtype=np.float32)

    fIN.close()
    fOUT.close()



if __name__ == '__main__':
    main()