#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:19:41 2019

Script to make query files for the ncedc

@author: amt
"""

from datetime import datetime, timedelta
from obspy.core import UTCDateTime
import subprocess
from obspy.clients.fdsn import Client
import multiprocessing

# smoke 'em if you got 'em
n_processor = multiprocessing.cpu_count()

data_dir = "../data"
client_list = ['NCEDC']
start_time = UTCDateTime("2018-01-01")
end_time = UTCDateTime("2019-01-03")
center_lon = -120.374
center_lat = 35.815
sta_radius = 0.5


station_list = {}
cl = client_list[0]
net = "BP"  # "*"
sta_wcard = "*"  #GHIB"
chan_wcard = "SH*,BH*,EH*,HH*,DP*,LH*"
inventory = Client(cl).get_stations(network=net, station=sta_wcard,
                                    channel=chan_wcard,
                                    latitude=center_lat, longitude=center_lon,
                                    maxradius=sta_radius, level='channel',
                                    includeavailability=True,
                                    starttime=start_time, endtime=end_time)

# Get list to download from invetory
sta_comp_list = {}
for net in inventory:
    net_str = net.code
    for sta in net:
        sta_str = sta.code
        for comp in sta:
            comp_list.append(comp.code)
        sta_comp_list[sta_str] = list(set(comp_list))


# Request
date_list = []
date = start_time._get_date()
while date + timedelta(days=1) < end_time:
    next_date = date + timedelta(days=1)
    date_list.append(date)
    date = next_date

def get_day(date):
    next_date = date + timedelta(days=1)
    doy = date.timetuple().tm_yday
    for sta in sta_comp_list.keys():
        for comp in sta_comp_list[sta_str]:
            file = str(date.year) + '_' + str(doy).zfill(3) + '_' + sta
            print(file)
            ## BK CMB -- BHZ 2010-03-25T00:00:00 2010-04-01T00:00:00
            tmp = f'NC {sta} * {comp} {date.isoformat()} {next_date.isoformat()}\n'
            print(tmp)
            # f = open('waveform.request', "w")
            # f.write(tmp)
            # f.close()
            # # curl --data-binary @waveform.request -o BK.miniseed http://service.ncedc.org/fdsnws/dataselect/1/query
            # command=f"curl --data-binary @waveform.request -o {file}.miniseed http://service.ncedc.org/fdsnws/dataselect/1/query"
            # subprocess.call(command, shell=True)

get_day(date_list[0])
# with multiprocessing.pool.ThreadPool(n_processor) as p:
#     p.map(get_day, date_list)
