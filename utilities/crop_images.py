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

    offset = 512
    start = 1000
    length = 350

    for file in files:
        print 'cropping file:\t\t', file
        crop_image(folderIN+file, folderOUT+file, start=start-offset, length=length)


def crop_image(fileIN, fileOUT, start, length):
    fIN = h5py.File(fileIN, "r")
    if 'wfs' not in fIN.keys(): raise AttributeError("wfs dataset not in file")
    fOUT = h5py.File(fileOUT, "w")

    wfs = np.asarray(fIN.get('wfs'))
    print 'changing length from:\t', wfs.shape[3], ' to ', length
    # del f['wfs']
    wfs = wfs[:, :, :, start:start + length]
    # print wfs.shape

    chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None
    print 'chunk size:\t', chunks_wfs

    fOUT.create_dataset("wfs", data=wfs, dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
    for key in fIN.keys():
        if key in ['wfs']: continue
        temp = np.asarray(fIN.get(key))
        fOUT.create_dataset(key, data=temp, dtype=np.float32)

    fIN.close()
    fOUT.close()



if __name__ == '__main__':
    main()