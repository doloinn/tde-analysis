# David Lynn
# Script which fits different function to light curves,
# including -5/3 index power law

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Curve fitting
from scipy.optimize import curve_fit
from scipy.odr.odrpack import *

# Gaia constants
cadence = [37, 26, 15]
ZEROFLUX = 3.631e-11  # W m-2 nm-1

# Helper functions to convert between flux and mag
def flux2mag(flux):
    return -2.5 * np.log10(flux / ZEROFLUX)

def mag2flux(mag):
    return ZEROFLUX * 10**(-mag / 2.5)

# Get tde dataframe
from getdfs import get_tde_df
tdedf = get_tde_df()

# Drop NaNs for detections
tdedf_detected = tdedf.dropna()

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

# Interpolated galaxy mags
galintermags = {0.5: [16.126666666666665, 18.120000000000001, 20.113333333333333], 2: [17.733333333333334, 19.699999999999999, 21.666666666666664], 5: [19.009999999999998, 20.670000000000002, 22.330000000000002]}
df = lc[0]

# Fitting functions
def decayfun(t, a, b, c):
    return a * (t + b)**(-5/3) + c
def linefun(x, a, b, c):
    return a * x + b
def expfun(x, a, b, c):
    return a * np.exp(-b * x) + c

# Put functions into list
funs = [decayfun, linefun, expfun]
funnames = ['Power law', 'Line', 'Exponential']

# Keeping track of outputs
fitdata = [[], [], []]
cntr = [0, 0, 0]

# Loop over light curves
for i in lc:

    # Galaxy brightness
    galmag = galintermags[i.iloc[0]['in galaxy radius']][int((i.iloc[0]['in galaxy mag'] - 16) / 2)]
    galflux = mag2flux(galmag)

    # Light curve brightnesses
    lcmags = i.dropna()['out found_mag'].as_matrix()
    lcfluxes = mag2flux(lcmags)
    lctimes = i.dropna()['in point time'].as_matrix()

    # Index of max
    peakidx = lcfluxes.argmax()

    # Loop over functions
    for j, fun in enumerate(funs):
        plt.figure()

        # ODR function takes different argument order
        odrfun = lambda B, x: fun(x, *B, galflux)
        tempfitdata = []
        plt.plot(lctimes, lcmags, '.')

        # Fit starting from max and starting from second from max
        for k in range(2):
            startidx = peakidx + k
            fittimes = lctimes[startidx:] - lctimes[startidx] 
            fitfluxes = lcfluxes[startidx:]

            # Try a standard curve fit to estimate parameters
            try:
                popt, pcov = curve_fit(lambda x, a, b: funs[j](x, a, b, galflux), fittimes, fitfluxes)
            except:
                popt = np.ones(2)

            # ODR
            fitfun = Model(odrfun)
            mydata = RealData(fittimes, fitfluxes)
            myodr = ODR(mydata, fitfun, beta0=popt)
            myoutput = myodr.run()

            # info <=3 means ODR converged
            if myoutput.info <= 3:
                tempfitdata.append(np.sqrt(myoutput.res_var) / lcfluxes[0])

            # Plotting
            fitplotx = np.linspace(fittimes.min(), fittimes.max(), 100)
            fitploty = odrfun(myoutput.beta, fitplotx)
            plt.plot(fitplotx + lctimes[startidx], flux2mag(fitploty), label=funnames[j])
        plt.xlabel('Time [days]')
        plt.ylabel('$G$ magnitude')
        plt.gca().invert_yaxis()

        # Grab the best fit or NaN
        if tempfitdata == []:
            cntr[j] += 1
            fitdata[j].append(np.nan)
        else:
            fitdata[j].append(min(tempfitdata))

# Dataframe from results
sourcedf = pd.DataFrame({'gal mag': [i.iloc[0]['in galaxy mag'] for i in lc], 'gal rad': [i.iloc[0]['in galaxy radius'] for i in lc], 'peak mag': [i.iloc[0]['in peak mag'] for i in lc], 'cadence': [i.iloc[1]['in point time'] - i.iloc[0]['in point time'] for i in lc]})
fitdf = pd.DataFrame({'powerlaw': fitdata[0], 'line': fitdata[1], 'exp': fitdata[2]})
outdf = pd.concat([fitdf, sourcedf], axis=1)

plt.show()
