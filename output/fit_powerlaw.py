# David Lynn
# Script which fits arbitrary powerlaw to light curves

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Imports for curve fitting
from scipy.optimize import curve_fit
from scipy.odr.odrpack import *

# Gaia constants
cadence = [37, 26, 15]
ZEROFLUX = 3.631e-11  # W m-2 nm-1

# Helper functions for converting between flux and mag
def flux2mag(flux):
    return -2.5 * np.log10(flux / ZEROFLUX)

def mag2flux(mag):
    return ZEROFLUX * 10**(-mag / 2.5)

# Get tde dataframe
from getdfs import get_tde_df
tdedf = get_tde_df()

# Drop NaNs for TDE detections
tdedf_detected = tdedf.dropna()

# Add light curve column to dataframe
tdedf['light curve'] = 0

# Sampled light curve
lc = []
for gm in tdedf['in galaxy mag'].unique():
    for gr in tdedf['in galaxy radius'].unique():
        for pm in tdedf['in peak mag'].unique():
            df = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
            for cad in cadence:
                for p in range(cad):
                    lidx = [p + i * cad for i in range(len(df) // cad)]
                    lc.append(df.iloc[lidx])

# Interpolated galaxy magnitudes
galintermags = {0.5: [16.126666666666665, 18.120000000000001, 20.113333333333333], 2: [17.733333333333334, 19.699999999999999, 21.666666666666664], 5: [19.009999999999998, 20.670000000000002, 22.330000000000002]}
df = lc[0]

# Powerlaw decay function
def decayfun(t, a, b, c, d):
    return a * (t + b)**c + d

# Empty list for powerlaw indices
pli = []

# Loop over light curves
for i in lc:

    # Galaxy brightness
    galmag = galintermags[i.iloc[0]['in galaxy radius']][int((i.iloc[0]['in galaxy mag'] - 16) / 2)]
    galflux = mag2flux(galmag)

    # Light curve brightnesses
    lcmags = i.dropna()['out found_mag'].as_matrix()
    lcfluxes = mag2flux(lcmags)
    lctimes = i.dropna()['in point time'].as_matrix()

    # Find index of max
    peakidx = lcfluxes.argmax()

    # ODR function takes different argument order
    odrfun = lambda B, x: decayfun(x, *B, galflux)
    temppli = []
    tempres = []

    # Fit to max and second from  max
    for k in range(2):
            startidx = peakidx + k

            # Get flux array and shift everything to zero star time
            fittimes = lctimes[startidx:] - lctimes[startidx] 
            fitfluxes = lcfluxes[startidx:]

            # Try standard curve_fit to get an estimate parameters before ODR
            try:
                popt, pcov = curve_fit(lambda x, a, b, c: decayfun(x, a, b, c, galflux), fittimes, fitfluxes)
            except:
                popt = np.ones(3)

            # ODR
            fitfun = Model(odrfun)
            mydata = RealData(fittimes, fitfluxes)
            myodr = ODR(mydata, fitfun, beta0=popt)
            myoutput = myodr.run()

            # info <= 3 means ODR converged, use output in this case
            if myoutput.info <= 3:
                temppli.append(myoutput.beta[2])
                tempres.append(myoutput.res_var)

    # Extract best fit power law index or NaN
    if len(temppli) > 0:
        pli.append(temppli[np.nanargmin(tempres)])
    else:
        pli.append(np.nan)

# Overall median
print('Total median: %s' %np.nanmedian(pli))

# Data frame with important parameters and indices
sourcedf = pd.DataFrame({'gal mag': [i.iloc[0]['in galaxy mag'] for i in lc], 'gal rad': [i.iloc[0]['in galaxy radius'] for i in lc], 'peak mag': [i.iloc[0]['in peak mag'] for i in lc], 'cadence': [i.iloc[1]['in point time'] - i.iloc[0]['in point time'] for i in lc]})
fitdf = pd.DataFrame({'powerlaw': pli})
outdf = pd.concat([fitdf, sourcedf], axis=1)

# Histogram results
for i in outdf.columns.drop('powerlaw', 1):
    for j in outdf[i].unique():
        meanval = outdf.loc[outdf[i] == j]['powerlaw'].mean()
        stdval = outdf.loc[outdf[i] == j]['powerlaw'].std()
        print('%s = %s,' %(i, j), 'mean: %s,' %meanval, 'standard deviation: %s,' %stdval, 'non-convergents: %s,' %(outdf.loc[outdf[i] == j]['powerlaw'].isnull().sum() / len(outdf.loc[outdf[i] == j]['powerlaw'])), outdf.loc[outdf[i] == j]['powerlaw'].isnull().sum(), len(outdf.loc[outdf[i] == j]['powerlaw']))
        plt.figure()
        plt.hist(outdf.loc[outdf[i] == j]['powerlaw'].dropna(), bins=20, ec='k')
        plt.axvline(-5./3., color='C3')
        plt.xlabel('Best fit $n$')
        plt.tight_layout()
        plt.gcf().canvas.set_window_title('%s=%s' %(i, j))

plt.show()
