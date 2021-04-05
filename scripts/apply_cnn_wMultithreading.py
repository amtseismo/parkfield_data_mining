#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run CNN in parallel on continuous data

@author: BGD

Based on apply_ccn.py from amt
"""
import obspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import datetime as dt
import glob
from os.path import join, basename
import os
import csv
import multiprocessing
from multiprocessing.pool import ThreadPool
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

now_str = dt.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
fh = logging.FileHandler(f'run_{now_str}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def simple_detrend(data):
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data


# Directories
data_dir = "../data"
model_weights_dir = "../pretrained_model_weights"
out_dir = "../cnn_output"

# set some params and options
plots = 0  # want figures
sampling_rate = 100  # set this to 100
shift = 15  # set this
thresh = 0.1
winlen = 15  # leave this
fac = 1
std = 0.2  # how long do you want the gaussian STD to be?
nwin = int(sampling_rate * winlen)  # leave this
nshift = int(sampling_rate * shift)  # leave this
epsilon = 1e-6  # leave this
# Hard coded input structure for 3 comp.
chan_keys = {'Z': 0, 'E': 1, '1': 1, 'N': 2, '2': 2}

# Use dropout layer?
drop = True

mainstart = dt.datetime.now()

# codestart = dt.datetime.now()
# import unet_tools
# # load model
# model_weights_name = "large_{:3.1f}_unet_lfe_std_{}.tf".format(fac, str(std))
#
# # BUILD THE MODEL
# print("Building the U-Net Archietecture ...")
# if drop:
#     model = unet_tools.make_large_unet_drop(fac, sampling_rate, ncomps=3)
#     model_weights_file = "drop_" + model_weights_name
# else:
#     model = unet_tools.make_large_unet(fac, sampling_rate, ncomps=3)
#
# # LOAD THE MODEL
# print('Loading training results from ' + model_weights_name)
# model.load_weights(join(model_weights_dir, model_weights_name))
#
# codestop = dt.datetime.now()
# runtime = (codestop - codestart).total_seconds() / 60
# logger.info("Loading the model took %s minutes", f"{runtime}")

# Get Stations in Data Folder
station_list = [basename(sta) for sta in glob.glob(join(data_dir, '*'))]
station_list = ['RAMR']
Nsta = len(station_list)
# Check what data is there

# smoke 'em if you got 'em!
max_processor = multiprocessing.cpu_count()

for sta_cnt, sta_str in enumerate(station_list):
    # Assume data has in the naming format: 'tchunk.network.station.channel.ms'
    files = glob.glob(join(data_dir, sta_str, '*.ms'))

    # Get unique time chunks of data
    tchunk_list = list(set([basename(f).split('.')[0] for f in files]))
    Nchunk = len(tchunk_list)

    # Directory to put all station output
    sta_out_dir = join(out_dir, f'{sta_str}')
    if not os.path.exists(sta_out_dir):
        os.makedirs(sta_out_dir)

    # Directory to put all those wonderful detections
    sta_csv_dir = join(sta_out_dir, 'CSV')
    if not os.path.exists(sta_csv_dir):
        os.makedirs(sta_csv_dir)

    # Directory to put all those beautiful figures
    if plots:
        sta_fig_dir = join(sta_out_dir, 'figures')
        if not os.path.exists(sta_fig_dir):
            os.makedirs(sta_fig_dir)

    logger.info('%s', f'** {sta_str} ** Station {sta_cnt+1} of {Nsta}')
    # loop over all station time-period datafiles - ussually days

    def process_timechunk(tchunk_str):
        codestart = dt.datetime.now()
        import unet_tools
        # load model
        model_weights_name = "large_{:3.1f}_unet_lfe_std_{}.tf".format(fac, str(std))

        # BUILD THE MODEL
        print("Building the U-Net Archietecture ...")
        if drop:
            model = unet_tools.make_large_unet_drop(fac, sampling_rate, ncomps=3)
            model_weights_file = "drop_" + model_weights_name
        else:
            model = unet_tools.make_large_unet(fac, sampling_rate, ncomps=3)

        # LOAD THE MODEL
        print('Loading training results from ' + model_weights_name)
        model.load_weights(join(model_weights_dir, model_weights_name))

        logger.info('Beginning to process %s', f'{sta_str}-{tchunk_str}')
        # CSV file to outout all detections
        csv_basename = f'{sta_str}-{tchunk_str}.csv'
        out_csv_fpath = join(sta_csv_dir, csv_basename)
        # If a detction file exits, skip it
        if os.path.isfile(out_csv_fpath):
            logger.info('Skipping this chunk, %s exists', csv_basename)
            return
        else:
            csv_file = open(out_csv_fpath, 'w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Station', 'Window Start', 'Detection_Time',
                                 'Detection_Amplitude'])
            csv_file.flush()

        # Find and read all components
        sta_wcard_str = join(data_dir, sta_str, f'{tchunk_str}*.ms')
        st = obspy.read(sta_wcard_str)
        assert len(st) == 3, "Must be 3 components!"

        # waveform processing
        st.detrend(type='simple')
        st.filter("highpass", freq=1.0, zerophase=True)
        logger.debug('Filtered %s', f'{sta_str}-{tchunk_str}')

        smin = smax = []
        for tr in st:
            smin.append(tr.stats.starttime)
            smax.append(tr.stats.endtime)
        # Pad With zero values for missing values
        start_time, end_time = min(smin), max(smax)
        st.trim(start_time, end_time, pad=True, fill_value=0)
        logger.debug('Trimmed %s', f'{sta_str}-{tchunk_str}')

        # Interpolate if sample-rate isnt 100 Hz
        for tr in st:
            if tr.stats.sampling_rate != sampling_rate:
                try:
                    tr.interpolate(sampling_rate=sampling_rate,
                                   starttime=start_time)
                except Exception:
                    logger.info("Inter. Failed, skipping %s",
                                f'{sta_str}-{tchunk_str}')

        # Apply CNN
        num_windows = (len(st[0]) - nwin) // nshift + 1
        logger.debug('Beginning t loop %s', f'{sta_str}-{tchunk_str}')

        # loop over all windows
        for win_cnt in range(num_windows):
            # Get index of current time window
            cur_idx = np.arange(win_cnt * nshift, win_cnt * nshift + nwin)
            # Concatonate them the in the same order no matter the order the
            # data was read into the stream
            snip = np.zeros(3 * nwin)
            for tr in st:
                chan_code = tr.stats.channel[-1]
                # First vert. then E, and N last
                chan_idx = np.arange(0, nwin) + nwin * chan_keys[chan_code]
                snip[chan_idx] = tr.data[cur_idx]

            # bit-wise normalization
            sign = np.sign(snip)
            # Envelope with no zeros
            val = np.log(np.abs(snip) + epsilon)
            cnninput = np.hstack([val[:1500].reshape(-1, 1),
                                  sign[:1500].reshape(-1, 1),
                                  val[1500:3000].reshape(-1, 1),
                                  sign[1500:3000].reshape(-1, 1),
                                  val[3000:].reshape(-1, 1),
                                  sign[3000:].reshape(-1, 1)])
            cnninput = cnninput[np.newaxis, :, :]
            # make s predictions
            stmp = model.predict(cnninput).ravel()
            # Detections must be 2 seconds apart
            spk = find_peaks(stmp, height=thresh, distance=200)

            # A peak is a detection, if we found one in this window, do stuff
            if len(spk[0] > 0):
                for peak_cnt in range(len(spk[0])):
                    # Get window startime
                    win_sec = win_cnt * nshift / sampling_rate
                    win_stime = start_time + dt.timedelta(seconds=win_sec)

                    # Convert from window sample index to time
                    peak_sec = spk[0][peak_cnt] / sampling_rate
                    peak_time = win_stime + dt.timedelta(seconds=peak_sec)
                    peak_amp = spk[1]['peak_heights'][peak_cnt]

                    # Write detection to file
                    csv_writer.writerow([sta_str, win_stime, f"{peak_time}",
                                         peak_amp])
                    csv_file.flush()

                    # Only plot if there is a detection and plots==True
                    if plots:
                        fig, ax = plt.subplots(4, 1, figsize=(6, 9))
                        nl = len(snip) // 3
                        for comp in range(3):
                            ax[comp].plot(snip[comp * nl:(comp + 1) * nl])
                        ax[3].plot(stmp, color=(0.25, 0.25, 0.25))
                        for peak in range(len(spk[0])):
                            ax[3].axvline(spk[0][peak], color='b')
                        ax[3].set_ylim((0, 1))
                        plt.savefig(f"Detection_{peak_time}")
                        plt.close()

        # Station runtime
        codestop = dt.datetime.now()
        runtime = (codestop - codestart).total_seconds() / 60
        logger.info("%s ran in %s minutes", f'{sta_str}-{tchunk_str}',
                    f"{runtime}")

    n_processor = min(Nchunk, max_processor)
    logger.info("Using %s processors to process %s", n_processor, Nchunk)
    with ThreadPool(n_processor) as p:
        p.map(process_timechunk, tchunk_list[0:2*n_processor])

    # for tchunk_str in tchunk_list:
    #     try:
    #         process_timechunk(tchunk_str)
    #     except Exception:
    #         logging.error("Error processing %s", f'{sta_str}-{tchunk_str}')

mainstop = dt.datetime.now()
runtime = (mainstop - mainstart).total_seconds() / 60
logger.info("\n \n The whole thing took took %s minutes", f"{runtime}")
