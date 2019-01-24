#!/usr/bin/env python

import numpy as np
import h5py
import argparse

import os
from os import listdir
from os.path import isfile,join

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-out', dest='folderOUT', help='folderOUT Path')
parser.add_argument('-in' , dest='folderIN' , help='folderIN Path')
#parser.add_argument('-file' , dest='filename' , help='folderIN Path')

args = parser.parse_args()
args.folderIN=os.path.join(args.folderIN,'')
args.folderOUT=os.path.join(args.folderOUT,'')

print
print 'Input Folder:\t', args.folderIN
print 'Output Folder:\t', args.folderOUT
print

files = [f for f in listdir(args.folderIN) if isfile(join(args.folderIN, f)) and '.hdf5' in f]

start, length = 0, 350 #1000, 350
slice = 1000
batchsize = 1
fletcher32 = True
shuffle = False
compression = ('gzip', 4)

counter = 0
for index, filename in enumerate(files):
	print "reading:\t", filename, 
											
	event_info_i = {}
	fIN = h5py.File(args.folderIN + str(filename), "r")
	for key in fIN.keys():
		if key in ['gains', 'wfs', 'wf_list']: continue
		event_info_i[key] = np.asarray(fIN[key])
	gains = np.asarray(fIN['gains'])
	wfs_i = np.asarray(fIN['wfs'])[:, :, start:start + length]
	fIN.close()

	wfs_i = np.asarray(wfs_i / gains[:, None])
	wfs_i = np.asarray(np.split(wfs_i, 4, axis=1))
	# wfs_i = np.asarray(np.split(wfs_i, 2, axis=1))
	wfs_i = np.swapaxes(wfs_i, 0, 1)
	#wfs_i = np.swapaxes(wfs_i, 1, 2) #ordering increases disk space by factor 1.8
	wfs_i = wfs_i[..., np.newaxis]
	if index==0:
		wfs = wfs_i
		event_info = event_info_i
	else:
		for key in event_info:
			event_info[key] = np.concatenate((event_info[key], event_info_i[key]))
		wfs = np.concatenate((wfs, wfs_i))

	print wfs.shape[0]
	while wfs.shape[0] >= slice:
		print "creating:\t", str(counter) + ".hdf5\t\t" , wfs.shape, "\t\t",

		chunks_wfs = (batchsize,) + wfs.shape[1:] if compression[0] is not None else None
		# chunks_wfs = None

		fOUT = h5py.File(args.folderOUT + str(counter) + ".hdf5", "w")
		dset1 = fOUT.create_dataset("wfs", data=wfs[:slice], dtype=np.float32, fletcher32=fletcher32, chunks=chunks_wfs, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
		dset = {}
		for key in event_info:
			# if key in ['EventNum']:
			# 	keytemp = 'MCEventNum'
			# elif key in ['RunNum']:
			# 	keytemp = 'MCRunNum'
			# else:
			# 	keytemp = key
			# dset[key] = fOUT.create_dataset(keytemp, data=event_info[key][:slice], dtype=np.float32, fletcher32=fletcher32, chunks=chunks, compression=compression[0], compression_opts=compression[1], shuffle=shuffle)
			dset[key] = fOUT.create_dataset(key, data=event_info[key][:slice], dtype=np.float32)
			event_info[key] = event_info[key][slice:]

		# ID = np.zeros((slice,), dtype=np.float32) # TODO for BKG events
		# ID = np.ones((slice,), dtype=np.float32) # TODO for SIG events
		# fOUT.create_dataset('ID', data=ID, dtype=np.float32)

		fOUT.close()

		counter += 1
		wfs = wfs[slice:]

		print wfs.shape[0], 'events remain'

print 'Events not considered:\t%d\t(%.2f %%)'%(wfs.shape[0], float(wfs.shape[0]*100.)/(wfs.shape[0]+counter*slice))
