#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:19:41 2019

Script to make query files for the ncedc

@author: amt
"""

from datetime import datetime, timedelta
from obspy.core import Stream, read, UTCDateTime
import subprocess

#stas=['GBB','GCK','GRO','GVV','KBU','KPR','LBR','LDL','LGO','LGY']
#comps=['EH?','EH?','EH?','EH?','EH?','EH?','HH?','HH?','EH?','HH?']
#stas=['KBU','KPR','LBR','LDL','LGO','LGY']
#comps=['EH?','EH?','HH?','HH?','EH?','HH?']

startdate=datetime(2019,7,1)
enddate=datetime(2019,12,11)

for ii in range(1):
    date=startdate+timedelta(days=ii)
    nextdate=date+timedelta(days=1)
    print(date)
    doy=date.timetuple().tm_yday
    for sta, cmp in zip(stas, comps):
        file=str(date.year)+'_'+str(doy).zfill(3)+'_'+sta
        print(file)       
        ## BK CMB -- BHZ 2010-03-25T00:00:00 2010-04-01T00:00:00
        tmp='NC '+sta+' * '+cmp+' '+date.isoformat()+' '+nextdate.isoformat()+'\n'
        print(tmp)        
        f = open('waveform.request', "w")
        f.write(tmp)
        f.close()
        # curl --data-binary @waveform.request -o BK.miniseed http://service.ncedc.org/fdsnws/dataselect/1/query
        command="curl --data-binary @waveform.request -o "+file+".miniseed http://service.ncedc.org/fdsnws/dataselect/1/query"
        subprocess.call(command, shell=True)