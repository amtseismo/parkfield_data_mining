#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:08:45 2020

Script to run CNN on continuous data

@author: amt
"""

from obspy.core import read, Stream
import numpy as np
from obspy import UTCDateTime
from datetime import timedelta
import matplotlib.pyplot as plt
import unet_tools
from scipy.signal import find_peaks
import pickle
import datetime

def simple_detrend(data):
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data

# set some params and options
plots=0 # want figures
sr=100 # set this to 100
shift=15 # set this
thresh=0.1 # minimum threshold to log detections
winlen=15 # leave this
drop=1 # drop layer
fac=1 # model size
std=0.2 # how long do you want the gaussian STD to be?
nwin=int(sr*winlen) # leave this
nshift=int(sr*shift) # leave this
epsilon=1e-6 # leave this

# set date and station
sta='THIS'
date=UTCDateTime(2018,8,3)
day=date.day

# MODEL NAME
model_save_file="large_"+'{:3.1f}'.format(fac)+"_unet_lfe_std_"+str(std)+".tf"

if drop:
    model_save_file="drop_"+model_save_file

# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=unet_tools.make_large_unet_drop(fac,sr,ncomps=3)
else:
    model=unet_tools.make_large_unet(fac,sr,ncomps=3)

# LOAD THE MODEL
print('Loading training results from '+model_save_file)
model.load_weights("./result_files/"+model_save_file)


print(sta+' '+str(day))
st=Stream()
st+=read("./data/2018-08-0"+str(day)+".BK."+sta+".HHZ.ms")
st+=read("./data/2018-08-0"+str(day)+".BK."+sta+".HHE.ms")
st+=read("./data/2018-08-0"+str(day)+".BK."+sta+".HHN.ms")
    
# process
st.detrend(type='simple')
st.filter("highpass", freq=1.0)

# print stats
start=st[0].stats.starttime
finish=st[0].stats.endtime

# cut traces so all components are the same start and end time
for ii in range(1,len(st)):
    if start<st[ii].stats.starttime:
        start=st[ii].stats.starttime
    if finish>st[ii].stats.endtime:
        finish=st[ii].stats.endtime
st.trim(starttime=start, endtime=finish,nearest_sample=1,pad=1,fill_value=0)

# interpolate if sr isnt 100
for tr in st:
    if sr != tr.stats.sampling_rate:
        tr.interpolate(sampling_rate=sr, starttime=start)
        
# define data
data0=st[0].data
data1=st[1].data
data2=st[2].data

# apply CNN to real data
nn=(len(data1)-nwin)//nshift

# intialize detection structure
sdects=[]
codestart=datetime.datetime.now()
maxvals=np.zeros(nn+1)
for ii in range(nn+1):
    # print(ii)
    data0s=simple_detrend(data0[ii*nshift:ii*nshift+nwin])
    data1s=simple_detrend(data1[ii*nshift:ii*nshift+nwin])
    data2s=simple_detrend(data2[ii*nshift:ii*nshift+nwin])
    snip=np.concatenate((data0s,data1s,data2s))
    sign=np.sign(snip)
    val=np.log(np.abs(snip)+epsilon)
    cnninput=np.hstack( [val[:1500].reshape(-1,1), sign[:1500].reshape(-1,1), val[1500:3000].reshape(-1,1), sign[1500:3000].reshape(-1,1), val[3000:].reshape(-1,1), sign[3000:].reshape(-1,1)] )
    cnninput=cnninput[np.newaxis,:,:]
    # make s predictions
    stmp=model.predict(cnninput)
    stmp=stmp.ravel()
    spk=find_peaks(stmp, height=thresh, distance=200)    
    maxvals[ii]=np.max(stmp)
    if plots:
        fig, ax = plt.subplots(4,1,figsize=(8,12))
        nl=len(snip)//3
        for jj in range(3):
            ax[jj].plot(snip[jj*nl:(jj+1)*nl])
        ax[3].plot(stmp,color=(0.25,0.25,0.25))
        for jj in range(len(spk[0])):
            ax[3].axvline(spk[0][jj],color='b') 
        ax[3].set_ylim((0,1))
    if len(spk[0]>0):     
        for kk in range(len(spk[0])):
            # sdects.append([spk[0][kk]+ii*nshift, st[0].times('timestamp')[spk[0][kk]+ii*nshift], spk[1]['peak_heights'][kk]]) #[stalat, stalon, spk[0][:]+ii*nshift, st[0].times('timestamp')[spk[0][:]+ii*nshift], 0])
            sdects.append([(spk[0][kk]+ii*nshift)/sr, spk[1]['peak_heights'][kk]])
codestop=datetime.datetime.now()
            
dects=np.asarray(sdects)            
file='picks_'+sta+'-'+str(day)+'.pkl'
with open(file, "wb") as fp:   #Pickling
    pickle.dump(dects, fp)
file='max_picks_'+sta+'-'+str(day)+'.pkl'
with open(file, "wb") as fp:   #Pickling
    pickle.dump(maxvals, fp)