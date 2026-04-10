#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft,ifft
import argparse
import datetime

# Source - https://stackoverflow.com/a/56530727
# Posted by janispritzkau
# Retrieved 2026-04-09, License - CC BY-SA 4.0
# Modified:
#  Now root raised cosine rather than raised cosine
#  Now in frequency domain
def rrcosfilter(N, beta, Ts, Fs):
    t = (np.arange(N) - N / 2) / Fs
    return np.sqrt(fft(np.where(np.abs(2*t) == Ts / beta,
        np.pi / 4 * np.sinc(t/Ts),
        np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts) ** 2))))

parser = argparse.ArgumentParser(
    prog = 'qpsk_doppler.py',
    description = 'Try to detect a QPSK signal through a Doppler scan in an SDRIQ file produced by SDRAngel')

parser.add_argument('filename')
parser.add_argument('--window_size',
                    type=int,
                    help='FFT window size',
                    default=65536)
parser.add_argument('--windows',
                    type=int,
                    help='Number of windows to process',
                    default=4096)
parser.add_argument('--windows_out',
                    type=int,
                    help='Number of windows to display',
                    default=16)
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
parser.add_argument('--bandpass',
                    help='Bandpass filter width in Hz; default is None',
                    type=int,
                    default=None)
parser.add_argument('--dopplersamples',
                    help='Number of Doppler samples',
                    type=int,
                    default=1024)
parser.add_argument('--dopplerstart',
                    help='Doppler search start frequency (Hz)',
                    type=float,
                    default=-100000)
parser.add_argument('--dopplerstop',
                    help='Doppler search stop frequency (Hz)',
                    type=float,
                    default=100000)

args = parser.parse_args()

window_size = args.window_size
windows = args.windows
windows_out = args.windows_out
offset = args.offset
nomarkers = args.nomarkers
tx_cf = args.center
bandpass = args.bandpass
dopplersamples = args.dopplersamples
dopplerstart = args.dopplerstart
dopplerstop = args.dopplerstop

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

    # Various frequency axes
    freq_axis = np.linspace(center_freq-sample_rate/2,
                            center_freq+sample_rate/2,
                            window_size)
    doppler_axis = np.linspace(dopplerstart,dopplerstop,dopplersamples)

    # Preallocate result array 
    doppler_samples = np.zeros((windows,dopplersamples),dtype=np.complex128)

    for i,dop in enumerate(doppler_axis):
        # Baseband the data
        data_baseband = data*np.exp(1j*2*np.pi*(tx_cf-center_freq-dop)/sample_rate*np.arange(window_size))

        # Apply RRC filter
        if bandpass is not None:
            data_baseband = ifft(fft(data_baseband,axis=1)*np.conjugate(rrcosfilter(window_size, 0.35, 1.0/bandpass, sample_rate)))

        # Squash the phase
        data_sq = data_baseband**4

        # Measure signal
        doppler_samples[:,i] = np.abs(np.mean(data_sq,axis=1))

    # Averaging
    data_smoothed = 20*np.log10(np.abs(ifft(fft(doppler_samples,axis=0),n=windows_out,axis=0)))
 
    # Display
    if windows_out == 1:
        # Peak detect
        detected_doppler = doppler_axis[np.argmax(data_smoothed,axis=1)].squeeze()

        plt.plot(doppler_axis,np.real(data_smoothed.squeeze()))
        plt.xlabel('Doppler frequency (Hz)')
        plt.ylabel('Relative signal level (dB)')
        plt.title('Detected Doppler {:.2f} Hz'.format(detected_doppler))
        plt.show()
    else:
        plt.imshow(np.real(data_smoothed),
                   extent=[dopplerstart,
                           dopplerstop,
                           offset*window_size/sample_rate,
                           (offset+windows)*window_size/sample_rate],
                   interpolation = 'none',
                   aspect = 'auto',
                   origin = 'lower'
                   )
        if not nomarkers:
            doppler_idx = np.argmax(data_smoothed,axis=1).squeeze()
            time_axis = np.linspace(offset*window_size/sample_rate,
                                    (offset+windows)*window_size/sample_rate,
                                    windows_out)
            doppler_dets = np.zeros_like(time_axis)
            for i,d in enumerate(doppler_idx):
                doppler_dets[i] = doppler_axis[d]

            plt.plot(doppler_dets,time_axis,'r+')
        plt.xlabel('Doppler frequency (Hz)')
        plt.ylabel('Time (s)')
        plt.show()
