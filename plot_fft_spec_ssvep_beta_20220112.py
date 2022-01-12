# -*- coding: utf-8 -*-
"""
Created on 12 Jan 2022

Plot FFT amplitude spectrum of SSVEP data
BETA dataset (Liu, B., et al. (2020). BETA: A large benchmark database toward SSVEP-BCI application. Frontiers in neuroscience, 14, 627.)
 
@author: Chi Man Wong
"""
import numpy as np
from numpy.matlib import repmat
from scipy import signal
import scipy.io 
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft


str_dir='../../../../Matlab/BETA SSVEP dataset/'
num_of_subj = 70

pha_val=repmat([0,0.5*np.pi,np.pi,1.5*np.pi],1,10)

sti_f=[8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8,
      10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8,
      12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8,
      14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 8.0, 8.2, 8.4]

target_order=np.argsort(sti_f)
sti_f=np.sort(sti_f)


Fs = 250
ch_used = [48,54,55,56,57,58,61,62,63]
oz_ch_idx = 7;
ch_used = [i - 1 for i in ch_used]

num_of_subbands=1                   # for filter bank analysis



latencyDelay=math.ceil(0.13*Fs)
dataLength = math.ceil(2*Fs)

fs = Fs
f0 = 50
Q = 35
notchB, notchA = signal.iircomb(f0, Q, ftype='notch', fs=fs)

fs = Fs/2
Nfft = 10*Fs
f = np.arange(0,Nfft,1)
f = f*Fs/Nfft
f_st = np.argmin(np.abs(f-4))
f_ed = np.argmin(np.abs(f-30))


bp_filterB = []
bp_filterA = []
for k in range(0, num_of_subbands):
    Wp = [8*(k+1)/(Fs/2),90/(Fs/2)]
    Ws = [(8*(k+1)-2)/(Fs/2),100/(Fs/2)]
    N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
    b, a = signal.cheby1(N, 0.5, Wn, 'bandpass')
    bp_filterB.append(b)
    bp_filterA.append(a)

for sn in range(0,num_of_subj):
    
    loaddata=scipy.io.loadmat(str_dir+'S'+str(sn+1)+'.mat')
    loaddata1=loaddata['data']
    loaddata2=loaddata1['EEG']
    eegdata=loaddata2[0,0]
    data1=np.transpose(eegdata,[0,1,3,2])
    eeg=data1[ch_used,math.floor(0.5*Fs):math.floor(0.5*Fs+latencyDelay)+dataLength,:,:]
    eeg=eeg[:,:,target_order,:]
    n_ch,n_point,n_sti,n_rep = eeg.shape
    
    subband_data = np.zeros((n_ch,dataLength,n_rep,n_sti,num_of_subbands))
    subband_ssvep_template = np.zeros((n_ch,dataLength,n_sti,num_of_subbands))
    for i in range(0,n_sti):
        for j in range(0,n_rep):
            y0 = eeg[:,:,i,j]            
            y1 = signal.filtfilt(notchB, notchA, y0, axis=1, padtype='odd', padlen=3*(max(len(notchB),len(notchA))-1))
            
            for sub_band in range(0,num_of_subbands):                
                y = signal.filtfilt(bp_filterB[sub_band], bp_filterA[sub_band], y1, axis=1, padtype='odd', padlen=3*(max(len(bp_filterB[sub_band]),len(bp_filterA[sub_band]))-1))
                subband_data[:,:,j,i,sub_band] = y[:,latencyDelay:latencyDelay+dataLength]
    
    fft_plot = np.zeros((n_sti,Nfft))
    for i in range(0,n_sti):
        for j in range(0,n_rep):
            x = subband_data[oz_ch_idx,:,j,i,0]
            X = np.abs(fft(x,Nfft))*2/Nfft
            fft_plot[i,:] = fft_plot[i,:] + X
        fft_plot[i,:] = fft_plot[i,:]/n_rep
    
    fig, ax = plt.subplots(nrows=5, ncols=8, sharex=True, sharey=True, dpi=150) #create figure handle
    for i in range(0,n_sti):
        if i<8:
            row_pos = 0
            col_pos = i            
        elif i<16:
            row_pos = 1
            col_pos = i-8
        elif i<24:
            row_pos = 2
            col_pos = i-16
        elif i<32:
            row_pos = 3
            col_pos = i-24
        elif i<40:
            row_pos = 4
            col_pos = i-32
        else:
            row_pos = 0
            col_pos = i
            
        ax[row_pos,col_pos].plot(f[f_st:f_ed],fft_plot[i,f_st:f_ed])    
        ax[row_pos,col_pos].set_xlabel('Frequency (Hz)')
        ax[row_pos,col_pos].set_ylabel('Amp.')
        
    fig.show()
    


