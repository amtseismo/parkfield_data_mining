#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to make query files for the ncedc in parallel

@author: amt, BGD
"""
import os
import tempfile
from datetime import datetime, timedelta
from obspy.core import UTCDateTime
import subprocess
from obspy.clients.fdsn import Client
import multiprocessing
from multiprocessing.pool import ThreadPool
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

now_str = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
fh = logging.FileHandler(f'data_pull_{now_str}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# smoke 'em if you got 'em
n_processor = multiprocessing.cpu_count()

# --------------------------------------------------------------------------- #
# Search and download parameters to change
out_dir = "/projects/amt/shared/parkfield/data"
client_list = ['NCEDC']
start_time = UTCDateTime("2018-01-01")
end_time = UTCDateTime("2019-01-01")
center_lon = -120.374
center_lat = 35.815
sta_radius = 1
cl = client_list[0]
net_str = "BP,BK,NC"  # "*"
sta_wcard = "*"
chan_wcard = "BH*,HH*,DP*,LH*"
# --------------------------------------------------------------------------- #

# Get station channel inventory
inventory = Client(cl).get_stations(network=net_str, station=sta_wcard,
                                    channel=chan_wcard,
                                    latitude=center_lat, longitude=center_lon,
                                    maxradius=sta_radius, level='channel',
                                    includeavailability=True,
                                    starttime=start_time, endtime=end_time)

# Get list to download from invetory
net_sta_comp_list = {}
for net in inventory:
    sta_comp_list = {}
    for sta in net:
        sta_out_dir = os.path.join(out_dir, f'{sta.code}')
        if not os.path.exists(sta_out_dir):
            os.makedirs(sta_out_dir)
        comp_list = []
        for comp in sta:
            comp_list.append(comp.code)
        if len(comp_list) == 3:
            sta_comp_list[sta.code] = list(set(comp_list))
    net_sta_comp_list[net.code] = sta_comp_list

 # Write to list
with open('net_sta_comp_list.json', 'w') as fp:
    json.dump(net_sta_comp_list, fp)

# Get list of dates to Request to batch out
date_list = []
date = start_time._get_datetime()
while date + timedelta(days=1) < end_time._get_datetime():
    print(date.isoformat())
    next_date = date + timedelta(days=1)
    date_list.append(date)
    date = next_date


def get_day(date):
    """
    Function which downloads a days of data for all station channels
    """
    next_date = date + timedelta(days=1)
    date_str = date.isoformat().split('T')[0]
    doy = date.timetuple().tm_yday
    for net_str, sta_comp_list in net_sta_comp_list.items():
        for sta_str, comp_list in sta_comp_list.items():
            for comp_str in comp_list:
                # Filename and path of datafile to write
                file_name = f"{date_str}.{net_str}.{sta_str}.{comp_str}.ms"
                file_path = os.path.join(out_dir, f"{sta_str}", file_name)
                logger.info(f"\n ** Trying: {file_name} **")

                # Check if file exists:
                if os.path.isfile(file_path):
                    logger.info(f"File exists, next please...")
                    continue
                else:
                    # Query request
                    tmp = f'{net_str} {sta_str} * {comp_str} {date.isoformat()} {next_date.isoformat()}\n'
                    logger.info(f"Query cmd: {tmp}")
                    scriptFile = tempfile.NamedTemporaryFile(delete=True)
                    http_str = "http://service.ncedc.org/fdsnws/dataselect/1/query"
                    with open(scriptFile.name, 'w') as f:
                        f.write(tmp)
                        f.close()
                        # curl --data-binary @waveform.request -o
                        # BK.miniseed http://service.ncedc.org/fdsnws/dataselect/1/query
                        cmd = f"curl --data-binary @{f.name} -o {file_path} {http_str}"
                        logger.info(f"curl call: {cmd}")
                        subprocess.call(cmd, shell=True)


# Batch it out in embarassingly parallel fashion - 1 day per cpu
with ThreadPool(max(n_processor, len(date_list))) as p:
    p.map(get_day, date_list)
