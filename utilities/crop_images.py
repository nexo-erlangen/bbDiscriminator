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
    mc_or_data = sys.argv[3]
    if not mc_or_data in ['mc', 'data']:
        raise ValueError('mc_or_data has wrong value: %s'%(str(mc_or_data)))
    if len(sys.argv)==5:
        mode = sys.argv[4]
        if not mode in ['Federico', 'None']:
            raise ValueError('mode must be Federico/None')
    else:
        mode = 'None'

    print
    print 'Input Folder:\t\t', folderIN
    print 'Output Folder:\t\t', folderOUT
    print 'Handling MC/Data:\t', mc_or_data
    if mode != 'None':
        print 'Mode:\t\t\t', mode
    print

    files = [f for f in listdir(folderIN) if isfile(join(folderIN, f)) and '.hdf5' in f]

    print 'Number of Files:\t', len(files)
    print

    offset = 0
    start = 1000
    length = 350

    # for i, file in enumerate(files):
    #     print 'cropping file:\t\t', file, 'to', str(i+18), '.hdf5'
    #     crop_image(folderIN+file, folderOUT+str(i+18)+'.hdf5', start=start-offset, length=length)
    #     # break

    for file in files:
        if isfile(folderOUT+file): continue
        print 'cropping file:\t\t', file
        crop_image(folderIN+file, folderOUT+file, start=start-offset, length=length, mc_or_data=mc_or_data, mode=mode)
        # break


def crop_image(fileIN, fileOUT, start, length, mc_or_data, mode):
    fIN = h5py.File(fileIN, "r")
    if 'wfs' not in fIN.keys(): raise AttributeError("wfs dataset not in file")
    fOUT = h5py.File(fileOUT, "w")

    if mc_or_data=='data':
        MC = 1024.8234
        data = np.asarray(fIN['APDTime'])
        start_i = np.rint(start-(MC-data)).astype(int)
        print start_i.shape
        print

    wfs = np.asarray(fIN.get('wfs'))
    print wfs.shape

    if mc_or_data=='data':
        print 'changing length from:\t', wfs.shape[2], ' to ', length

        wfs_temp = np.zeros((wfs.shape[0], wfs.shape[1], length), dtype=np.float32)
        for i in xrange(wfs.shape[0]):
            try:
                # wfs_temp[i] = wfs[i, :, :, start_i[i]:start_i[i] + length]
                wfs_temp[i] = wfs[i, :, start_i[i]:start_i[i] + length]
            except:
                print fileIN, 'does not work'
                print i, start_i[i], length, MC, data[i]
                print wfs.shape, wfs_temp.shape
                return
            if i%1000==0:
                print i, wfs_temp[i].shape
        print wfs.shape
        print wfs_temp.shape
        wfs = wfs_temp
        print wfs.shape
    elif mc_or_data=='mc':
        wfs = wfs[:, :, :, start:start + length]
    else:
        raise ValueError('mc_or_data has wrong value: %s'%(str(mc_or_data)))

    chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None
    print 'chunk size:\t', chunks_wfs

    fOUT.create_dataset("wfs", data=wfs, dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in fIN.keys():
        if key in ['wfs']: continue
        temp = np.asarray(fIN.get(key))
        if mode=='Federico':
            if key in ['LXeEnergy', 'ID', 'APDEnergy', 'QValue']: continue
            if key in ['MCEventNumber']: key = 'EventNumber'
            elif key in ['MCRunNumber']: key = 'RunNumber'
        fOUT.create_dataset(key, data=temp, dtype=np.float32)

    if mode=='Federico':
        fIN_NumEvents = len(np.asarray(fIN.get('MCEventNumber')))
        if mc_or_data=='data':
            for key in ['MCEnergy', 'MCPosU', 'MCPosV', 'MCPosX', 'MCPosY', 'MCPosZ']:
                if key in fIN.keys(): continue
                temp = np.zeros((fIN_NumEvents,), dtype=np.float32)
                fOUT.create_dataset(key, data=temp, dtype=np.float32)
            IsMC = np.zeros((fIN_NumEvents,), dtype=np.float32)
        elif mc_or_data=='mc':
            IsMC = np.ones((fIN_NumEvents,), dtype=np.float32)
        else:
            raise ValueError('mc_or_data has wrong value: %s' % (str(mc_or_data)))
        fOUT.create_dataset('IsMC', data=IsMC, dtype=np.float32)

    fIN.close()
    fOUT.close()



if __name__ == '__main__':
    main()
