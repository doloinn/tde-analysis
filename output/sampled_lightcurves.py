# David Lynn
# Script which samples simulated light curves according to Gaia's
# cadence and plots the result

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Get tde dataframe
from getdfs import get_tde_df
tdedf = get_tde_df()
tdedf_detected = tdedf.dropna()

# Gaia cadences
cadence = [37, 26, 15]
lc = []
tdedf['light curve'] = 0

# Sampling
for gm in tdedf['in galaxy mag'].unique():
    for gr in tdedf['in galaxy radius'].unique():
        for pm in tdedf['in peak mag'].unique():
            df = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
            for cad in cadence:
                for p in range(cad):
                    lidx = [p + i * cad for i in range(len(df) // cad)]
                    lc.append(df.iloc[lidx])

# Plot random sample of 10 light curves
for i in random.sample(lc, 10):
    df_det = i.dropna()
    gr = i.iloc[0]['in galaxy radius']
    gm = i.iloc[0]['in galaxy mag']
    pm = i.iloc[0]['in peak mag']

    fig = plt.figure()
    plt.plot(i['in point time'], i['total source_g_mag'], 'b.', label='Simulated')
    plt.plot(df_det['in point time'], df_det['out found_mag'], 'rx', label='Detected')
    # plt.axhline(i.iloc[0]['galaxy source_g_mag'], label='Galaxy magnitude', color='purple')
    # plt.axhline(20.7, label='Gaia cut off', color='green')
    plt.xlabel('Time [days]')
    plt.ylabel('Brightness [ $G$ mag]')
    # plt.title('Galaxy radius: %s, galaxy V magnitude: %s, TDE peak magnitude: %s' % (gr, gm, pm))
    plt.gca().invert_yaxis()
    plt.legend(loc='best')
    fig.canvas.set_window_title('%s, %s, %s' %(gr, gm, pm))

plt.show()
