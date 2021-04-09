#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to make query files for the ncedc in parallel

@author: amt, BGD
"""
import os
from datetime import datetime, timedelta
from obspy.core import UTCDateTime
import subprocess
from obspy.clients.fdsn import Client
import multiprocessing
from multiprocessing.pool import ThreadPool
# smoke 'em if you got 'em
n_processor = multiprocessing.cpu_count()

out_dir = "../data"
client_list = ['NCEDC']
start_time = UTCDateTime("2018-01-01")
end_time = UTCDateTime("2019-01-01")
center_lon = -120.374
center_lat = 35.815
sta_radius = 0.5


station_list = {}
cl = client_list[0]
net_str = "BP"  # "*"
sta_wcard = "*"  #GHIB"
chan_wcard = "SH*,BH*,EH*,HH*,DP*,LH*"
inventory = Client(cl).get_stations(network=net_str, station=sta_wcard,
                                    channel=chan_wcard,
                                    latitude=center_lat, longitude=center_lon,
                                    maxradius=sta_radius, level='channel',
                                    includeavailability=True,
                                    starttime=start_time, endtime=end_time)

# Get list to download from invetory
sta_comp_list = {}
for net in inventory:
    net_code = net.code
    for sta in net:
        sta_out_dir = os.path.join(out_dir, f'{sta.code}')
        if not os.path.exists(sta_out_dir):
            os.makedirs(sta_out_dir)
        comp_list = []
        for comp in sta:
            comp_list.append(comp.code)
        if len(comp_list)==3:
            sta_comp_list[sta.code] = list(set(comp_list))


# Request
date_list = []
date = start_time._get_datetime()
while date + timedelta(days=1) < end_time._get_datetime():
    print(date.isoformat())
    next_date = date + timedelta(days=1)
    date_list.append(date)
    date = next_date

def get_day(date):
    next_date = date + timedelta(days=1)
    date_str = date.isoformat().split('T')[0]
    doy = date.timetuple().tm_yday
    for sta_str, comp_list in sta_comp_list.items():
        for comp_str in comp_list:
            file_name = f"{date_str}.{net_str}.{sta_str}.{comp_str}.ms"
            file_path = os.path.join(f"../data/{sta_str}", file_name)
            print(file_path)
            ## BK CMB -- BHZ 2010-03-25T00:00:00 2010-04-01T00:00:00
            tmp = f'{net_str} {sta_str} * {comp_str} {date.isoformat()} {next_date.isoformat()}\n'
            print(tmp)
            f = open('waveform.request', "w")
            f.write(tmp)
            f.close()
            # curl --data-binary @waveform.request -o BK.miniseed http://service.ncedc.org/fdsnws/dataselect/1/query
            command=f"curl --data-binary @waveform.request -o {file_path} http://service.ncedc.org/fdsnws/dataselect/1/query"
            subprocess.call(command, shell=True)

with ThreadPool(n_processor) as p:
    p.map(get_day, date_list)
