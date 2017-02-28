import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get galaxy dataframes
from getdfs import get_galaxy_dfs
eldf, spdf = get_galaxy_dfs()

# Drop NaNs for detected
eldf_detected = eldf.dropna()
spdf_detected = spdf.dropna()

# Plot for ellipticals 
# uni = np.concatenate([eldf['source_g_mag'].unique(), eldf_detected['out found_mag'].unique()])
# bins = np.linspace(min(uni), max(uni), 25)
# plt.hist(eldf['source_g_mag'], bins=bins, color='C2', edgecolor='k')
# plt.hist(eldf_detected['source_g_mag'], bins=bins, color='C0', edgecolor='k')
# plt.hist(eldf_detected['out found_mag'], bins=bins, color='C3', edgecolor='k', alpha=0.6)
# uni = np.concatenate([spdf['source_g_mag'].unique(), spdf_detected['out found_mag'].unique()])
# bins = np.linspace(min(uni), max(uni), 25)
# plt.hist(spdf['source_g_mag'], bins=bins, color='C2', edgecolor='k')
# plt.hist(spdf_detected['source_g_mag'], bins=bins, color='C0', edgecolor='k')
# plt.hist(spdf_detected['out found_mag'], bins=bins, color='C3', edgecolor='k', alpha=0.6)
# plt.xlabel('$G$ magnitude')

# Plot spirals separated
f, axarr = plt.subplots(3, 1, sharex=True)

for i, r in enumerate(spdf['in disk radius'].unique()):
    magdf = spdf.loc[spdf['in disk radius'] == r]
    magdf_detected = magdf.dropna()
    uni = np.concatenate([magdf['source_g_mag'].unique(), magdf['out found_mag'].unique()])
    # spc = 12
    bins = 50
    bins = np.linspace(min(uni), max(uni), bins)

    axarr[i].hist(magdf['source_g_mag'], bins=bins, color='C2', edgecolor='black')
    axarr[i].hist(magdf_detected['source_g_mag'], bins=bins, color='C0', edgecolor='black')
    for c, ba in enumerate(spdf['in b/t'].unique()):
        axarr[i].hist(magdf_detected.loc[spdf['in b/t'] == ba]['out found_mag'], bins=bins, color='C%s' %(c+3), edgecolor='black', alpha=0.6, label='$B/T = %s$' %ba)

    axarr[i].set_title("Radius $= %s''$" %r)

legstuff = axarr[2].get_legend_handles_labels()
axarr[2].set_xlabel('$G$ magnitude')
plt.figlegend(legstuff[0][2:], legstuff[1][2:], loc='best')
plt.tight_layout()

plt.show()