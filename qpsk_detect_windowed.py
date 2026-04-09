#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
import argparse
import datetime

parser = argparse.ArgumentParser(
    prog = 'qpsk_detect_windowed.py',
    description = 'Try to detect a QPSK signal in an SDRIQ file produced by SDRAngel')

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
parser.add_argument('--nomarkers',
                    help='Disable markers on the display',
                    action = 'store_true',
                    default=False)
parser.add_argument('--center',
                    help='Desired center frequency in Hz',
                    type=int,
                    default=2216.5e6)
parser.add_argument('--fftsamples',
                    help='Number of samples in Doppler FFT',
                    type=int,
                    default=1024)
parser.add_argument('--bandpass',
                    help='Bandpass filter width in Hz; default is None',
                    type=int,
                    default=None)
parser.add_argument('--agcsize',
                    help='Size of AGC window',
                    type=int,
                    default=1)

args = parser.parse_args()

window_size = args.window_size
windows = args.windows
fftsamples = args.fftsamples
offset = args.offset
nomarkers = args.nomarkers
tx_cf = args.center
bandpass = args.bandpass
agc_size = args.agcsize

symbol_rates = [72e3, 2e6, 4e6, 6e6] # Possible symbol rates

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
                       count=2*window_size*windows)
    
    # Unpack the data into the proper complex type
    data = np.reshape(data, (windows,window_size,2), order='C')
    data = data[:,:,0] + 1j*data[:,:,1]
    freq_axis = np.linspace(center_freq-sample_rate/2,
                            center_freq+sample_rate/2,
                            window_size)

    # Use a bandpass filter on the signal
    if bandpass is not None:
        data = ifft(fft(data,axis=1)*np.conjugate(np.abs(freq_axis-tx_cf)<(bandpass/2)),axis=1)

    # Baseband the data
    data = data*np.exp(1j*2*np.pi*(tx_cf-center_freq)/sample_rate*np.arange(window_size))

    # AGC the data
    data = data / ifft(fft(np.abs(data),n=agc_size,axis=1),n=window_size,axis=1)

    # Squash the phase
    data_sq = data**4
    
    # FFT
    data_fft = fft(data_sq,axis=1,n=fftsamples)

    # Averaging
    data_smoothed = 20*np.log10(ifft(fft(np.abs(data_fft),axis=0),n=1,axis=0))
    freq_axis=np.linspace(0,sample_rate,fftsamples)

    # Peak detect
    det_symbol_rate = freq_axis[np.argmax(data_smoothed.squeeze())]

    # Display
    plt.plot(freq_axis,np.real(data_smoothed.squeeze()),'b')
    if not nomarkers:
        plt.plot([det_symbol_rate,det_symbol_rate],plt.ylim(),'c')
        for sr in symbol_rates:
            plt.plot([sr,sr],plt.ylim(),'r')
    plt.plot(freq_axis,np.real(data_smoothed.squeeze()),'b')
    plt.xlabel('Doppler frequency (Hz)')
    plt.ylabel('Relative signal level (dB)')
    plt.title('Detected Doppler rate : {} kHz'.format(det_symbol_rate/1e3))
    plt.show()
    
    
