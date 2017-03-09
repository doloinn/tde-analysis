# David Lynn
# Script which plots unsampled simulated light curves

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get tde dataframe
from getdfs import get_tde_df
tdedf = get_tde_df()

# Drop NaNs for detected sources
tdedf_detected = tdedf.dropna()

# Whole light curve plots
for gm in tdedf['in galaxy mag'].unique()[:]:
    for gr in tdedf['in galaxy radius'].unique()[:]:
        for pm in tdedf['in peak mag'].unique()[:]:
            plotdf = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
            plotdf_detected = tdedf_detected.loc[tdedf_detected['in galaxy mag'] == gm].loc[tdedf_detected['in galaxy radius'] == gr].loc[tdedf_detected['in peak mag'] == pm]
            x = plotdf['in point time']
            y = plotdf['total source_g_mag']
            x_det = plotdf_detected['in point time']
            y_det = plotdf_detected['out found_mag']
            plt.figure()
            plt.plot(x, y, '.', alpha=0.2, label='Simulated')
            plt.plot(x_det, y_det, 'xC3', alpha=0.2, label='Detected')
            plt.xlabel('Time [days]')
            plt.ylabel('Brightness [$G$ mag]')
            plt.gca().invert_yaxis()
            plt.legend(loc='best')
            fig = plt.gcf()
            fig.canvas.set_window_title('%s, %s, %s' %(gr, gm, pm))

plt.show()
