# -*- coding: utf-8 -*-
"""
Created on 12 Jan 2022

Plot FFT amplitude spectrum of SSVEP data
OpenBMI dataset (Lee, M. H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy. GigaScience, 8(5), giz002.)

@author: Chi Man Wong
"""
import numpy as np
from scipy import signal
import scipy.io 
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft



str_dir='//10.119.68.246/BCIShare/Database/SSVEP_Resting_EEG_dataset/'
num_of_subj = 54

pha_val=[0,0,0,0]
sti_f=[12.00,8.57,6,67,5,45]

Fs = 250
ch_used = [25,28,61,44,62,32,29,30,31]
ch_used = [i - 1 for i in ch_used]
oz_ch_idx = 7;

num_of_trials=25
num_of_subbands=1                   # for filter bank analysis


latencyDelay=math.ceil(0.13*Fs)
dataLength = math.ceil(3.5*Fs)

fs = Fs
f0 = 50
Q = 35
notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=fs)

fs = Fs/2

Nfft = 10*Fs
f = np.arange(0,Nfft,1)
f = f*Fs/Nfft
f_st = np.argmin(np.abs(f-1))
f_ed = np.argmin(np.abs(f-40))


bp_filterB = []
bp_filterA = []
for k in range(0, num_of_subbands):
    Wp = [4*(k+1)/(Fs/2),40/(Fs/2)]
    Ws = [(4*(k+1)-2)/(Fs/2),50/(Fs/2)]
    N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
    b, a = signal.cheby1(N, 0.5, Wn, 'bandpass')
    bp_filterB.append(b)
    bp_filterA.append(a)

for sn in range(0,num_of_subj):    
    if sn<9:
        loaddata=scipy.io.loadmat(str_dir+'s'+str(sn+1)+'/'+'sess01_subj0'+str(sn+1)+'_EEG_SSVEP.mat')
    else:
        loaddata=scipy.io.loadmat(str_dir+'s'+str(sn+1)+'/'+'sess01_subj'+str(sn+1)+'_EEG_SSVEP.mat')
    loaddata1=loaddata['EEG_SSVEP_train']
    loaddata2a=loaddata1['smt']
    loaddata2b=loaddata1['y_dec']
    eegdata=loaddata2a[0,0]
    eegdata=eegdata[0:-1:4,:,ch_used]
    eegdata_label=loaddata2b[0,0]
    
    mycount=np.zeros((1,4),dtype=int)
    n_ch = eegdata.shape[2]
    n_totalpoint= eegdata.shape[0]
    n_totaltrial= eegdata.shape[1]
    eeg = np.zeros((n_ch,n_totalpoint,4,25))
    for tn in range(0,n_totaltrial):
        x0 = eegdata[:,tn,:]      
        y0 = eegdata_label[0,tn]-1
        eeg[:,:,y0,mycount[0,y0]]= x0.T
        mycount[0,y0]=mycount[0,y0]+1   
    
    
    n_ch,n_point,n_sti,n_rep = eeg.shape
    
    subband_data = np.zeros((n_ch,dataLength,n_rep,n_sti,num_of_subbands))
    subband_ssvep_template = np.zeros((n_ch,dataLength,n_sti,num_of_subbands))
    for i in range(0,n_sti):
        for j in range(0,n_rep):
            y0 = eeg[:,:,i,j]            
            y1 = signal.filtfilt(notchB, notchA, y0, axis=1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))
            
            for sub_band in range(0,num_of_subbands):
                #y = signal.filtfilt(bp_filterB[sub_band], bp_filterA[sub_band], y1, axis=1)
                y = signal.filtfilt(bp_filterB[sub_band], bp_filterA[sub_band], y1, axis=1, padtype='odd', padlen=3*(max(len(bp_filterB[sub_band]),len(bp_filterA[sub_band]))-1))
                subband_data[:,:,j,i,sub_band] = y[:,latencyDelay:latencyDelay+dataLength]
    
    fft_plot = np.zeros((n_sti,Nfft))
    for i in range(0,n_sti):
        for j in range(0,n_rep):
            x = subband_data[oz_ch_idx,:,j,i,0]
            X = np.abs(fft(x,Nfft))*2/Nfft
            fft_plot[i,:] = fft_plot[i,:] + X
        fft_plot[i,:] = fft_plot[i,:]/n_rep
    
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, dpi=150) #create figure handle
    for i in range(0,n_sti):
        ax[i].plot(f[f_st:f_ed],fft_plot[i,f_st:f_ed])    
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].set_ylabel('Amp.')
        
    fig.show()
    


