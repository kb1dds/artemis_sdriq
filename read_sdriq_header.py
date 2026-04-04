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
    sample_rate = np.fromfile(fp, dtype='uint32', count=1, sep='')
    center_freq = np.fromfile(fp, dtype='uint64', count=1, sep='')
    timestamp = np.fromfile(fp, dtype='uint64', count=1, sep='')
    sample_size = np.fromfile(fp, dtype='uint32', count=1, sep='')
    crc = np.fromfile(fp, dtype='uint32', count=1, sep='', offset=1)

    print('Sample rate {} Hz'.format(sample_rate[0]))
    print('Center freq {} Hz'.format(center_freq[0]))
    print('Sample size {} bits'.format(sample_size[0]))
    print('Timestamp {}'.format(datetime.datetime.fromtimestamp(timestamp[0]/1000,datetime.UTC).isoformat()))
