#!/bin/python

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import torch
from torch.fft import fft,ifft
import argparse
import datetime
import tqdm

def doppler_processor(data,sample_rate,center_freq,tx_cf,doppler_axis,filt=None,device='cpu'):
    '''
    Run Doppler sweep for QPSK signal detection

    Inputs:
       data         = complex samples (windows,window_size)
       sample_rate  = I/Q sample rate in Hz
       center_freq  = center frequency of sampler in Hz
       tx_cf        = expected transmitter center frequency in Hz
       doppler_axis = Doppler frequencies to try (vector, Hz)
       filt         = filter shape to apply (window_size)
       device       = torch device for computation

    Outputs:
       doppler_samples = estimate of signal at given Doppler (windows,doppler_axis length)
    '''
    windows = data.shape[0]
    window_size = data.shape[1]
    doppler_size = doppler_axis.shape[0]

    doppler_samples = torch.zeros((windows,doppler_size),dtype=torch.complex128)
    
    for i,dop in enumerate(tqdm.tqdm(doppler_axis,desc='Doppler sweep',position=2,leave=False)):
        # Apply filtering if requested
        if filt is not None:
            data_baseband = ifft(fft(data,axis=1)*filt)
        else:
            data_baseband = data

        # Baseband the data
        data_baseband = data_baseband*torch.exp(1j*2*torch.pi*(tx_cf-center_freq-dop)/sample_rate*torch.arange(window_size).to(device))

        # Squash the phase
        data_sq = data_baseband**4

        # Measure signal
        doppler_samples[:,i] = torch.abs(torch.mean(data_sq,axis=1))
        
    return doppler_samples            

# Source - https://stackoverflow.com/a/56530727
# Posted by janispritzkau
# Retrieved 2026-04-09, License - CC BY-SA 4.0
# Modified:
#  Now root raised cosine rather than raised cosine
#  Now in frequency domain
def rrcosfilter(N, beta, Ts, Fs):
    t = (torch.arange(N) - N / 2) / Fs
    return torch.sqrt(fft(torch.where(torch.abs(2*t) == Ts / beta,
        torch.pi / 4 * torch.sinc(t/Ts),
        torch.sinc(t/Ts) * torch.cos(torch.pi*beta*t/Ts) / (1 - (2*beta*t/Ts) ** 2))))

parser = argparse.ArgumentParser(
    prog = 'qpsk_doppler.py',
    description = 'Performance validation for Doppler QPSK signal detection')

parser.add_argument('--window_size',
                    type=int,
                    help='FFT window size',
                    default=65536)
parser.add_argument('--windows',
                    type=int,
                    help='Number of windows to process',
                    default=256)
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
                    default=256)
parser.add_argument('--dopplerstart',
                    help='Doppler search start frequency (Hz)',
                    type=float,
                    default=-100000)
parser.add_argument('--dopplerstop',
                    help='Doppler search stop frequency (Hz)',
                    type=float,
                    default=100000)
parser.add_argument('--device',
                    help='PyTorch device',
                    default='cpu')
parser.add_argument('--outfile',
                    help='Name of file to save figure to (default is to show figure)',
                    default=None)
parser.add_argument('--snrstart',
                    help='SNR to start sweep (dB)',
                    type=float,
                    default=-20)
parser.add_argument('--snrstop',
                    help='SNR to stop sweep (dB)',
                    type=float,
                    default=5)
parser.add_argument('--snrsamples',
                    help='Number of SNR values to test',
                    type=int,
                    default=10)
parser.add_argument('--sample_rate',
                    help='Sample rate Hz',
                    type=int,
                    default=10e6)
parser.add_argument('--symbol_rate',
                    help='Symbol rate Hz',
                    type=int,
                    default=4e6)
parser.add_argument('--center_freq',
                    help='Center frequency in Hz',
                    type=int,
                    default=2.2615e9)
parser.add_argument('--true_dop',
                    help='True Doppler frequency in Hz',
                    type=int,
                    default=-12e3)

args = parser.parse_args()

window_size = args.window_size
windows = args.windows
tx_cf = args.center
bandpass = args.bandpass
dopplersamples = args.dopplersamples
dopplerstart = args.dopplerstart
dopplerstop = args.dopplerstop
outfile = args.outfile
device = args.device
snrstart = args.snrstart
snrstop = args.snrstop
snrsamples = args.snrsamples
sample_rate = args.sample_rate
center_freq = args.center_freq
symbol_rate = args.symbol_rate
true_dop = args.true_dop

# Which processing window sizes counts to test
windows_list = [2**i for i in range(8,14)]

# Which doppler frequencies to test
doppler_axis = np.linspace(dopplerstart,dopplerstop,dopplersamples)
true_doppler_sample = int(np.min(np.where(doppler_axis-true_dop>0.)))
snr_axis = np.linspace(snrstart,snrstop,snrsamples)

# Preallocate output
doppler_estimates = np.zeros((snrsamples,len(windows_list)))
doppler_snr = np.zeros((snrsamples,len(windows_list)))

for wi,window_size in enumerate(tqdm.tqdm(windows_list,'Running window sizes',position=0)):
    # Preallocate result array for each Doppler processing run 
    doppler_samples = torch.zeros((windows,dopplersamples),dtype=torch.complex128)

    # Pre-build bandpass filter if needed
    if bandpass is not None:
        filt=torch.conj(rrcosfilter(window_size, 0.35, 1.0/bandpass, sample_rate).to(device))
    else:
        filt = None

    # Synthesize fixed signal
    time_axis = np.linspace(0,windows*window_size/sample_rate,windows*window_size)
    message_axis = np.arange(0,windows*window_size/sample_rate,1/symbol_rate)
    message = np.floor(np.random.rand(*message_axis.shape)*4)/2
    message = np.exp(1j*np.pi*scipy.interpolate.interpn((message_axis,),message,time_axis,method='nearest',bounds_error=False,fill_value=0.))
    signal = np.exp(-1j*(2*np.pi*(tx_cf-true_dop-center_freq)*time_axis)+message)
    signal_power = np.var(signal)

    for i,snr in enumerate(tqdm.tqdm(snr_axis,desc='Running SNR samples',position=1,leave=False)):
        # Noise profile (AWGN)
        noise = np.random.randn(*time_axis.shape)
        noise_power = np.var(noise)

        # Synthesize data
        data = 10**(snr/20.)*np.sqrt(noise_power/signal_power)*signal+noise
    
        # Unpack the data into the proper shape
        data = np.reshape(data, (windows,window_size), order='C')
        data = torch.from_numpy(data).to(device)

        # Apply Doppler processing
        doppler_samples = doppler_processor(data,sample_rate,center_freq,tx_cf,doppler_axis,filt,device)

        # Averaging
        data_smoothed = 20*torch.log10(torch.abs(torch.mean(doppler_samples,axis=0))).to('cpu').numpy()
    
        # Doppler detection
        doppler_estimates[i,wi] = doppler_axis[np.argmax(data_smoothed)].squeeze()
        doppler_snr[i,wi] = np.real(np.max(data_smoothed[true_doppler_sample-2:true_doppler_sample+2]) - np.mean(data_smoothed[np.abs(doppler_axis)>true_dop]) - np.std(data_smoothed[np.abs(doppler_axis)>true_dop])).squeeze()

#        plt.plot(doppler_axis,np.real(data_smoothed.squeeze()))
#        plt.plot([true_dop,true_dop],plt.ylim())
#        plt.plot(doppler_axis[true_doppler_sample],np.real(data_smoothed[true_doppler_sample]),'b+')
#        plt.xlabel('Doppler frequency (Hz)')
#        plt.ylabel('Relative signal level (dB)')
#        plt.title('Detected Doppler {:.2f} Hz, SNR {:.2f} dB'.format(doppler_estimates[wi,i],doppler_snr[wi,i]))
#        plt.show()

# Display
plt.plot(snr_axis,doppler_snr)
plt.legend(windows_list)
plt.xlabel('Signal SNR (dB)')
plt.ylabel('Doppler estimate SNR (dB)')

# Save or display output figure
if outfile is None:
    plt.show()
else:
    plt.savefig(outfile)
