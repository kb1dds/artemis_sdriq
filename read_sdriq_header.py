#!/bin/python

import numpy as np
import argparse
import datetime

parser = argparse.ArgumentParser(
    prog = 'read_sdriq_header.py',
    description = 'Read the header from an SDRIQ file produced by SDRAngel')

parser.add_argument('filename')

args = parser.parse_args()

with open(args.filename,'rb') as fp:
    sample_rate = np.fromfile(fp, dtype='uint32', count=1, sep='')[0]
    center_freq = np.fromfile(fp, dtype='uint64', count=1, sep='')[0]
    timestamp = np.fromfile(fp, dtype='uint64', count=1, sep='')[0]
    sample_size = np.fromfile(fp, dtype='uint32', count=1, sep='')[0]
    crc = np.fromfile(fp, dtype='uint32', count=1, sep='',offset=4)[0]
    lo_freq = center_freq - sample_rate/2

    # Assess number of samples
    data_start = fp.tell()
    fp.seek(0,2)
    if sample_size == 16:
        num_samples = (fp.tell()-data_start)/4
    elif sample_size == 24: # NB apparently actually "24" means 32 bits
        num_samples = (fp.tell()-data_start)/8

    print('Timestamp:      {}'.format(datetime.datetime.fromtimestamp(timestamp/1000,datetime.UTC).isoformat()))
    print('Sample rate:    {} Hz'.format(sample_rate))
    print('Center freq:    {} Hz'.format(center_freq))
    print('Local osc freq: {} Hz'.format(lo_freq))
    print('Sample size:    {} bits'.format(sample_size))
    print('Sample count:   {}'.format(num_samples))
    print('Length:         {:.2f} s'.format(num_samples/sample_rate.astype('float64')))
    
    
