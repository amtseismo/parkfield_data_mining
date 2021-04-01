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
thresh=0.1
winlen=15 # leave this
drop=1
fac=1
std=0.2 # how long do you want the gaussian STD to be?
nwin=int(sr*winlen) # leave this
nshift=int(sr*shift) # leave this
epsilon=1e-6 # leave this

# load model
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

for day in np.arange(3,4):
    for sta in ['B079','40','THIS','B901','THIS','TSCN','TCHL']:
        print(sta+' '+str(day))
        st=Stream()
        if sta=='40':
            st=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DPZ.2018-08-0"+str(day)+"-00-00-00.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DP1.2018-08-0"+str(day)+"-00-00-00.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DP2.2018-08-0"+str(day)+"-00-00-00.ms")
        elif sta=="THIS" or sta=="TCHL" or sta=="TSCN":
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHZ.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHE.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHN.ms")
        elif sta=="B079" or sta=="B901":
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EHZ.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EH1.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EH2.ms")    
        elif sta=="PAGB":
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".NC."+sta+".HHZ.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".NC."+sta+".HHE.ms")
            st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".NC."+sta+".HHN.ms")
            
        # process
        st.detrend(type='simple')
        st.filter("highpass", freq=1.0)
        
        # print stats
        print("Trace length is "+str(len(st)))
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

# # # if plots:
# # #     # load PNSN stuff to compare
# # #     # PNSN Catalog format: 0Evid,1Magnitude,2Magnitude Type,3Epoch(UTC),4Time UTC,5Time Local,6Distance From,7Lat,8Lon,9Depth Km,10Depth Mi
# # #     # regional events from pnsn catalog
# # #     # PNSN Catalog format: 0Evid,1Magnitude,2Magnitude Type,3Epoch(UTC),4Time UTC,5Time Local,6Distance From,7Lat,8Lon,9Depth Km,10Depth Mi
# # #     pnsn_cat=np.genfromtxt("pnsn_event_export_20200205.csv", delimiter=",", skip_header=1, usecols=(0,1,4,8,9,10), dtype=("i8", float, "|U19", float, float, float))
# # #     df= pd.DataFrame(pnsn_cat, columns=["ID","Date","Time","Magnitude","Lat","Lon","Depth","Regional"])
# # #     for ii in range(len(pnsn_cat)):
# # #       print(str(ii)+" of "+str(len(pnsn_cat)))
# # #       df= df.append({"ID": pnsn_cat[ii][0], "Magnitude": pnsn_cat[ii][1], "Date": pnsn_cat[ii][2][:10], "Time": pnsn_cat[ii][2][11:],
# # #                     "Lat": pnsn_cat[ii][3], "Lon": pnsn_cat[ii][4], "Depth": pnsn_cat[ii][5], "Regional": "uw"}, ignore_index=True)
# # #     df['day'] = pd.DatetimeIndex(df['Date']).day
# # #     df=df[df['day']==st[0].stats.starttime.day]
    
# # #     # distance calculation
# # #     def distance(origin, destination):
# # #         lat1, lon1 = origin
# # #         lat2, lon2 = destination
# # #         radius = 6371 # km
    
# # #         dlat = math.radians(lat2-lat1)
# # #         dlon = math.radians(lon2-lon1)
# # #         a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
# # #             * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
# # #         c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
# # #         d = radius * c
    
# # #         return d
    
# # #     # add distances to dataframe
# # #     dists=np.zeros(len(df))
# # #     for ii in range(len(df)):
# # #         dists[ii]=distance([47.8554, -121.9710],[df.iloc[ii]['Lat'],df.iloc[ii]['Lon']])
# # #     df['distance']=dists
    
# # #     # load regional velocity model and cacluate p and s wave travel times
# # #     dists, depths, parv, sarv = pickle.load( open( "p4.pkl", "rb" ) )
# # #     e3p=interp2d(dists,depths,parv)
# # #     e3s=interp2d(dists,depths,sarv)
# # #     ps=np.zeros(len(df))
# # #     ss=np.zeros(len(df))
# # #     for ii in range(len(df)):
# # #         ps[ii]=e3p(df.iloc[ii]['distance'],df.iloc[ii]['Depth'])
# # #         ss[ii]=e3s(df.iloc[ii]['distance'],df.iloc[ii]['Depth'])
# # #     df['ps']=ps
# # #     df['ss']=ss
    
# # #     # make plots
# # #     fig, ax = plt.subplots(nrows=3, ncols=1,sharex=True)
# # #     times=st[0].times('relative')
# # #     inc=100
# # #     d0n=np.max(np.abs(data0))
# # #     d1n=np.max(np.abs(data1))
# # #     d2n=np.max(np.abs(data2))
# # #     ax[0].plot(times,data0/d0n,color=(0.5,0.5,0.5))
# # #     ax[1].plot(times,data1/d1n,color=(0.5,0.5,0.5))
# # #     ax[2].plot(times,data2/d2n,color=(0.5,0.5,0.5))
# # #     for ii in range(len(pdects)):
# # #         ax[0].plot(times[pdects[ii]-inc:pdects[ii]+inc],data0[pdects[ii]-inc:pdects[ii]+inc]/d0n,color=(0.5,0,0),alpha=0.5)
# # #         ax[1].plot(times[pdects[ii]-inc:pdects[ii]+inc],data1[pdects[ii]-inc:pdects[ii]+inc]/d1n,color=(0.5,0,0),alpha=0.5)
# # #         ax[2].plot(times[pdects[ii]-inc:pdects[ii]+inc],data2[pdects[ii]-inc:pdects[ii]+inc]/d2n,color=(0.5,0,0),alpha=0.5)
# # #     for ii in range(len(sdects)):
# # #         ax[0].plot(times[sdects[ii]-inc:sdects[ii]+inc],data0[sdects[ii]-inc:sdects[ii]+inc]/d0n,color=(0,0,0.5),alpha=0.5)
# # #         ax[1].plot(times[sdects[ii]-inc:sdects[ii]+inc],data1[sdects[ii]-inc:sdects[ii]+inc]/d1n,color=(0,0,0.5),alpha=0.5)
# # #         ax[2].plot(times[sdects[ii]-inc:sdects[ii]+inc],data2[sdects[ii]-inc:sdects[ii]+inc]/d2n,color=(0,0,0.5),alpha=0.5)
# # #     for ii in range(len(df)):
# # #         eqortime=UTCDateTime(int(df.iloc[ii]['Date'][:4]),int(df.iloc[ii]['Date'][5:7]),int(df.iloc[ii]['Date'][8:10]),int(df.iloc[ii]['Time'][:2]),int(df.iloc[ii]['Time'][3:5]),int(df.iloc[ii]['Time'][6:8]))
# # #         parv=eqortime-st[0].stats.starttime+df.iloc[ii]['ps']
# # #         sarv=eqortime-st[0].stats.starttime+df.iloc[ii]['ss']
# # #         for a in ax:
# # #             a.axvline(parv,color=(0,0.5,0),alpha=0.25)
# # #             a.axvline(sarv,color=(0,0.5,0),alpha=0.25)
# # #     ax[0].set_ylim((-0.01,0.01))
# # #     ax[1].set_ylim((-0.01,0.01))
# # #     ax[2].set_ylim((-0.01,0.01))