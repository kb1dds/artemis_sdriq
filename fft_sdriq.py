#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import argparse
import datetime

parser = argparse.ArgumentParser(
    prog = 'fft_sdriq_header.py',
    description = 'Run averaged FFT on SDRIQ file produced by SDRAngel')

parser.add_argument('filename')
parser.add_argument('--window_size',
                    type=int,
                    help='FFT window size',
                    default='1024')
parser.add_argument('--windows',
                    type=int,
                    help='Number of windows to process',
                    default='1')

args = parser.parse_args()

window_size = int(args.window_size)
windows = int(args.windows)

with open(args.filename,'rb') as fp:
    sample_rate = np.fromfile(fp, dtype='uint32', count=1, sep='')[0]
    center_freq = np.fromfile(fp, dtype='uint64', count=1, sep='')[0]
    timestamp = np.fromfile(fp, dtype='uint64', count=1, sep='')[0]
    sample_size = np.fromfile(fp, dtype='uint32', count=1, sep='')[0]
    crc = np.fromfile(fp, dtype='uint32', count=1, sep='',offset=4)[0]

    # Assess number of samples
    data_start = fp.tell()
    fp.seek(0,2)
    if sample_size == 16:
        sample_dtype = 'int16'
        num_samples = (fp.tell()-data_start)/4
    elif sample_size == 24: # NB apparently actually "24" means 32 bits
        sample_dtype = 'int32'
        num_samples = (fp.tell()-data_start)/8
    fp.seek(data_start,0)

    # Pull data from the file
    data = np.fromfile(fp,
                       dtype=sample_dtype,
                       sep='',
                       count=2*windows*window_size)

    # Unpack the data into the proper complex type
    data = np.reshape(data, (2,windows,window_size))
    data = data[0,:,:] + 1j*data[1,:,:]
    data.squeeze()

    # FFT and averaging
    data_fft = np.sum(fft(data,axis=1), axis=0)

    # Frequency axis
    freq_axis = np.linspace(center_freq-sample_rate/2,
                            center_freq+sample_rate/2,
                            window_size)

    # Display
    plt.plot(freq_axis,10*np.log10(np.abs(data_fft)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative signal level (dB)')
    plt.show()
    
    
    
