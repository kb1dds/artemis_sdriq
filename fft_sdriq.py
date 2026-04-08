#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
import argparse
import datetime

parser = argparse.ArgumentParser(
    prog = 'fft_sdriq_header.py',
    description = 'Run averaged FFT on SDRIQ file produced by SDRAngel')

parser.add_argument('filename')
parser.add_argument('--window_size',
                    type=int,
                    help='FFT window size',
                    default=1024)
parser.add_argument('--windows',
                    type=int,
                    help='Number of windows to process',
                    default=1)
parser.add_argument('--offset',
                    type=int,
                    help='Number of windows to skip before processing',
                    default=0)
parser.add_argument('--windows_out',
                    type=int,
                    help='Number of windows to display in plot',
                    default=1)
parser.add_argument('--equalize',
                    help='Equalize the output by swapping left/right halves',
                    action = 'store_true',
                    default=False)
parser.add_argument('--nomarkers',
                    help='Disable markers on the display',
                    action = 'store_true',
                    default=False)

args = parser.parse_args()

window_size = args.window_size
windows = args.windows
offset = args.offset
windows_out = args.windows_out
equalize = args.equalize
nomarkers = args.nomarkers

# Target signal parameters
tx_cf = 2216.5e6 # Center frequency (Hz)
tx_bw = 2e6      # Bandwidth (Hz)
tx_dop = -18e3   # Doppler frequency offset (Hz)

with open(args.filename,'rb') as fp:
    sample_rate = np.fromfile(fp, dtype='uint32', count=1, sep='')[0]
    center_freq = np.fromfile(fp, dtype='uint64', count=1, sep='')[0]
    timestamp = np.fromfile(fp, dtype='uint64', count=1, sep='')[0]
    sample_size = np.fromfile(fp, dtype='uint32', count=1, sep='')[0]
    crc = np.fromfile(fp, dtype='uint32', count=1, sep='',offset=4)[0]

    # Frequency axis
    freq_axis = np.linspace(center_freq-sample_rate/2,
                            center_freq+sample_rate/2,
                            window_size)

    # Assess number of samples
    data_start = fp.tell()
    fp.seek(0,2)
    if sample_size == 16:
        sample_bytes = 4
        sample_dtype = 'int16'
        num_samples = (fp.tell()-data_start)/4
    elif sample_size == 24: # NB apparently actually "24" means 32 bits
        sample_bytes = 8
        sample_dtype = 'int32'
        num_samples = (fp.tell()-data_start)/8
        
    # Pull data from the file
    fp.seek(data_start+window_size*offset*sample_bytes,0)
    data = np.fromfile(fp,
                       dtype=sample_dtype,
                       sep='',
                       count=2*windows*window_size)

    # Unpack the data into the proper complex type
    data = np.reshape(data, (windows,window_size,2), order='C')
    data = data[:,:,0] + 1j*data[:,:,1]
    data.squeeze()

    # FFT
    data_fft = fft(data,axis=1)

    # Averaging
    #data_smoothed = 20*np.log10(ifft(fft(np.abs(data_fft),axis=0),n=windows_out,axis=0))
    data_smoothed = 20*np.log10(data_fft)

    # Equalization if requested
    if equalize:
        #baseline = 20*np.log10(np.mean(np.abs(data_fft),axis=0))
        baseline = data_smoothed
        baseline = (baseline+np.roll(baseline[:,::-1],axis=1,shift=1))/2
        data_smoothed = data_smoothed-baseline

    # Display
    if windows_out == 1:
        plt.plot(freq_axis,np.real(data_smoothed.squeeze()))
        if not nomarkers:
            plt.plot([tx_cf,tx_cf],plt.ylim(),'r:')
            plt.plot([tx_cf+tx_dop,tx_cf+tx_dop],plt.ylim(),'b:')
            plt.plot([tx_cf-tx_bw/2,tx_cf-tx_bw/2],plt.ylim(),'r')
            plt.plot([tx_cf+tx_bw/2,tx_cf+tx_bw/2],plt.ylim(),'r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Relative signal level (dB)')
        plt.show()
    else:
        plt.imshow(np.real(data_smoothed),
                   extent=[center_freq-sample_rate/2,
                           center_freq+sample_rate/2,
                           offset*window_size/sample_rate,
                           (offset+windows)*window_size/sample_rate],
                   interpolation = 'none',
                   aspect = 'auto',
                   origin = 'lower'
                   )
        if not nomarkers:
            plt.plot([tx_cf,tx_cf],plt.ylim(),'r:')
            plt.plot([tx_cf+tx_dop,tx_cf+tx_dop],plt.ylim(),'b:')
            plt.plot([tx_cf-tx_bw/2,tx_cf-tx_bw/2],plt.ylim(),'r')
            plt.plot([tx_cf+tx_bw/2,tx_cf+tx_bw/2],plt.ylim(),'r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Time (s)')
        plt.show()
    
    
