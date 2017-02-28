import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get tde dataframe
from getdfs import get_tde_df
tdedf = get_tde_df()

# Drop NaNs for detected
tdedf_detected = tdedf.dropna()

# Plot separated detected source and detected mags
f, axarr = plt.subplots(3, 1, sharex=True)

for i, r in enumerate(tdedf['in galaxy radius'].unique()):
    magdf = tdedf.loc[tdedf['in galaxy radius'] == r]
    magdf_detected = magdf.dropna()
    uni = np.concatenate([magdf['total source_g_mag'].unique(), magdf['out found_mag'].unique()])
    bins = 50
    bins = np.linspace(min(uni), max(uni), bins)

    axarr[i].hist(magdf['total source_g_mag'], bins=bins, color='C2', edgecolor='black')
    axarr[i].hist(magdf_detected['total source_g_mag'], bins=bins, color='C0', edgecolor='black')
    for c, ba in enumerate(tdedf['in galaxy mag'].unique()):
        axarr[i].hist(magdf_detected.loc[tdedf['in galaxy mag'] == ba]['out found_mag'], bins=bins, color='C%s' %(c+3), edgecolor='black', alpha=0.6, label='$V_G = %s$' %ba)

    axarr[i].set_title("Radius $= %s''$" %r)

legstuff = axarr[2].get_legend_handles_labels()
axarr[2].set_xlabel('$G$ magnitude')
plt.figlegend(legstuff[0][2:], legstuff[1][2:], loc='best')
plt.tight_layout()

plt.show()