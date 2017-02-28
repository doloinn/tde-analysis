# David Lynn
# Code to generate TDE simulator input sample

# Directory stuff
import os
import shutil

# Output data to pickle file
import pickle

# Standard stuff
import numpy as np
import matplotlib.pyplot as plt

# Constants
ZEROFLUX = 3.631e-11  # W m-2 nm-1
NOMAG = 27  # mags
PEAKMAGS = [14, 16, 18]  # mags
TIMES_OBSERVED = [50, 70, 120]
DECAY_START_TIME = 20  # days
END_TIME = 10  # days
SMOOTHING_FACTOR = 2
RISE_STEEPNESS = 1
TIME_STEP = 1  # days
FINAL_TIME = 500  # days

# Gaia cadence
cadence = [5 * 365.25 / i for i in TIMES_OBSERVED]

# Time before TDE set to largest cadence so at least one point will be
# observed before disruption
time_before_tde = np.ceil(max(cadence))

# Galaxy magnitudes and radii
galaxymags = [16, 18, 20]
galaxyradii = [0.5, 2, 5]

# TDE blackbody spectrum that simulator requires
gibis_string = "watts 300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000,1010,1020,1030,1040,1050,1060,1070,1080,1090,1100|4901053.38782,4530937.95778,4190959.54612,3878908.4638,3592617.2126,3330003.53593,3089096.59785,2868051.25869,2665154.07932,2478823.69234,2307607.44333,2150175.66435,2005314.54564,1871918.28196,1748980.96026,1635588.50344,1530910.87582,1434194.6786,1344756.20859,1261975.01514,1185287.96447,1114183.80269,1048198.19779,986909.234051,929933.328432,876921.537018,827556.2195,781548.030429,738633.207492,698571.128847,661142.113572,626145.441324,593397.569361,562730.527041,533990.469787,507036.376244,481738.873974,457979.180498,435648.147858,414645.400078,394878.55402,376262.51512,358718.840376,342175.161767,326564.664015,311825.6112,297900.917373,284737.756763,272287.209679,260503.94059,249345.905239,238774.083971,228752.238747,219246.69157,210226.122294,201661.383963,193525.334062,185792.68017,178439.838702,171444.805516,164787.037325,158447.342921,152407.783322,146651.580071,141163.03093,135927.432349,130931.008099,126160.843543,121604.825052,117251.584139,113090.4459,109111.381398,105304.963676,101662.327073,98175.1295951,94835.5180799,91636.0959226,88569.8931707,85630.3387865,82811.2349104,80106.7329663"

# Directory and pickle file name
dir_name = 'tdegen'
pickle_file = 'tdes.pickle'


# Make file name from index
def path_to_file(i):
    file_name = 'gibis_tde'
    return os.path.join(dir_name, '%s_%s.pix' % (file_name, i))

# Convert flux to magnitude
def flux2mag(flux):
    return -2.5 * np.log10(flux / ZEROFLUX)

# Convert magnitude to flux
def mag2flux(mag):
    return ZEROFLUX * 10**(-mag / 2.5)

# Function of light curve
def lightcurve_function(x, A, B, C, D):
    '''Derivative of - A * np.tanh(B * (C + x)) + D'''
    return D * x - (A * np.log(np.cosh(B * (C + x)))) / B

# Actual light curve
def lightcurve(peakmag, time, B=2, D=5):
    logtime = np.log(time / 365.25)
    desired_centre = np.log(DECAY_START_TIME / 365.25)
    A = D + 5 / 3
    C = np.arctanh(D / A) / B - desired_centre
    logflux = lightcurve_function(logtime, A, B, C, D) - lightcurve_function(desired_centre, A, B, C, D)
    flux = mag2flux(peakmag) * np.exp(logflux)
    return flux

# Pre-disruption flux
noflux = mag2flux(27)

# Delete and create directory of outputs
Delete and create directory before starting
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)
os.makedirs(dir_name)

# Layout in grid
gridx = list(range(-1600, 2001, 200))
gridy = list(range(-500, 501, 500))
xpoints = len(gridx)
ypoints = len(gridy)

# Initialise indices and dictionary of sources
file_index = 1
source_index = 0
object_dict = {}

# Open file first time
pixfile = open(path_to_file(file_index), 'a')

# Loop over peak magnitudes
for peak in PEAKMAGS:
    # Time and flux for disruption sampled daily
    time_tde = np.arange(TIME_STEP, FINAL_TIME, TIME_STEP)
    flux_tde = lightcurve(peak, time_tde, SMOOTHING_FACTOR, RISE_STEEPNESS)

    # Time and flux before disruption sampled daily
    time_pre = np.arange(-time_before_tde, time_tde[0], TIME_STEP)
    flux_pre = np.zeros(len(time_pre))

    # Total time and flux and magnitudes
    time = np.concatenate([time_pre, time_tde])
    flux = np.concatenate([flux_pre, flux_tde]) + noflux
    mag = flux2mag(flux)

    # Initialise times and magnitudes
    sampled_times = []
    sampled_mags = []

    # Loop over galaxy magnitudes and radii
    for galmag in galaxymags:
        for galrad in galaxyradii:

            # Initialise source index and arrays
            i = 0
            sampled_times = []
            sampled_mags = []

            # Loop over daily sampled magnitudes
            for idx, k in enumerate(mag):
                i += 1
                sampled_times.append(time[idx])
                sampled_mags.append(mag[idx])

                # Iterate over grid locations
                ra = gridx[(source_index // (2 * ypoints)) % xpoints]
                dec = gridy[(source_index // 2) % ypoints]

                # Galaxy parameters
                v_minus_i = 1.39
                bulge_total_ratio = 1
                minor_major_ratio = 1
                position_angle = 0
                z = 0.01

                # Dictionary of galaxy data
                object_dict[source_index] = {'object': 'galaxy', 'data': {'ra': ra, 'dec': dec, 'galaxy mag': galmag, 'galaxy radius': galrad}}

                # String to be input to simulator
                galaxystring = '202 [%s] %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g 0 0 0 %.5g\n' % (source_index, ra, dec, galmag, v_minus_i, bulge_total_ratio, galrad, minor_major_ratio, position_angle, z)
                pixfile.write(galaxystring)

                # TDE is new source
                source_index += 1
                object_dict[source_index] = {'object': 'tde', 'data': {'peak mag': peak, 'snapshot mag': mag[idx], 'point time': time[idx], 'ra': ra, 'dec': dec}}
                writestring = '901 [%s] %.5g %.5g %.5g %s\n' % (source_index, ra, dec, mag[idx], gibis_string)
                pixfile.write(writestring)

                # New file when grid is full
                source_index += 1
                if source_index % (xpoints * ypoints) == 0:
                    file_index += 1
                    pixfile.close()
                    pixfile = open(path_to_file(file_index), 'a')

    # Plot sample light curves
    plt.figure()
    plt.plot(time, flux2mag(flux), label='TDE lightcurve')
    # plt.semilogx(time, flux2mag(flux), '.', label='TDE lightcurve')
    # plt.xticks([1e-12, 1e3])
    plt.xlabel('Time [days]')
    plt.ylabel('Brightness in G [mags]')
    plt.gca().invert_yaxis()
    plt.savefig('lightcurve.png')

# Dumping the source data to pickle files
with open(pickle_file, 'wb') as f:
    pickle.dump(object_dict, f)

plt.show()

