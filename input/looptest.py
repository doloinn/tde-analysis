
# import numpy as np
import random

import os
import shutil
import pickle

dir_name = 'galaxygen'
pickle_file = 'galaxies.pickle'


def path_to_file(i):
    file_name = 'gibis_galaxy'
    return os.path.join(dir_name, '%s_%s.pix' % (file_name, i))


# ELLIPTICALS

bulge_total_ratio = 1  # ellipticals
number_per_object = 500

# GIBIS parameters
minor_major_ratio_list = [0.3, 0.65, 1]  # motivate
radius_list = [0.5, 2, 5]  # too small # actually probably OK (based on de Souza) but motivate
v_mag_list = [15, 18, 21]  # OK

# v_minus_i_range_ell = (-3, 0)  # choose better
v_minus_i_range_ell = (1.15, 1.6)  # choose better
position_angle_range = (0, 360)  # OK
# z_range = (0.001, 0.1)  # 170--600 Mpc optimal distance according to de Souza

# Delete and create directory before starting
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)

os.makedirs(dir_name)

# Layout in grid
gridx = list(range(-1600, 2001, 200))
gridy = list(range(-500, 501, 500))
xpoints = len(gridx)
ypoints = len(gridy)


# ra = [gridx[(source_index // ypoints) % xpoints] for source_index in range(1000)]
# dec = [gridy[source_index % ypoints] for source_index in range(1000)]
# print(ra, dec)
# exit()


# Initialise loop indices
file_index = 1
source_index = 0
object_dict = {}

# Open file first time
pixfile = open(path_to_file(file_index), 'a')

for minor_major_ratio in minor_major_ratio_list:
    for radius in radius_list:
        for v_mag in v_mag_list:
            for i in range(number_per_object):
                v_minus_i = random.uniform(*v_minus_i_range_ell)
                position_angle = random.uniform(*position_angle_range)
                # z = random.uniform(*z_range)
                z = 0.01
                ra = gridx[(source_index // ypoints) % xpoints]
                dec = gridy[source_index % ypoints]
                # ra = gridx[source_index // xpoints % ypoints]
                # dec = gridy[source_index % xpoints]


                object_dict[source_index] = {'galaxy type': 'elliptical', 'data': {'ra': ra, 'dec': dec, 'v mag': v_mag, 'v-i': v_minus_i, 'radius': radius, 'b/a': minor_major_ratio, 'theta': position_angle}}

                writestring = '202 [%s] %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g 0 0 0 %s\n' % (source_index, ra, dec, v_mag, v_minus_i, bulge_total_ratio, radius, minor_major_ratio, position_angle, z)
                pixfile.write(writestring)

                source_index += 1
                if source_index % (xpoints * ypoints) == 0:
                    file_index += 1
                    pixfile.close()
                    pixfile = open(path_to_file(file_index), 'a')

if source_index % (xpoints * ypoints) == 0:
    pixfile.close()
    file_index += 1

# SPIRALS

bulge_total_ratio_list = [0.1, 0.4, 0.7]
number_per_object = 500
radius_disk_bulge_ratio = 3
minor_major_ratio_disk = 1

# GIBIS parameters
minor_major_ratio_list = [0.3, 0.65, 1]  # motivate
# radius_list = [0.1, 1, 5]  # too small # actually probably OK (based on de Souza) but motivate
# v_mag_list = [15, 18, 21]  # OK

v_minus_i_range_sp = (0.95, 1.4)  # choose better
position_angle_range = (0, 360)  # OK
# z_range = (0.001, 0.1)  # 170--600 Mpc optimal distance according to de Souza

vmi_m = (1.386 - 1.161) / (1 - 0.244)
vmi_c = 1.386 - vmi_m
vmi = lambda bt: vmi_m * bt + vmi_c

number_per_object = 100
# source_index = 0
pixfile = open(path_to_file(file_index), 'a')

for minor_major_ratio_bulge in minor_major_ratio_list:
    for radius_disk in radius_list:
        for v_mag in v_mag_list:
                for bulge_total_ratio in bulge_total_ratio_list:
                    for i in range(number_per_object):
                        v_minus_i = random.uniform(*v_minus_i_range_sp) - 1.16 + vmi(bulge_total_ratio)
                        position_angle = random.uniform(*position_angle_range)
                        # z = random.uniform(*z_range)
                        z = 0.01
                        ra = gridx[(source_index // ypoints) % xpoints]
                        dec = gridy[source_index % ypoints]
                        radius_bulge = radius_disk / radius_disk_bulge_ratio
                        
                        object_dict[source_index] = {'galaxy type': 'spiral', 'data': {'ra': ra, 'dec': dec, 'v mag': v_mag, 'v-i': v_minus_i, 'b/t': bulge_total_ratio, 'bulge radius': radius_bulge, 'bulge b/a': minor_major_ratio_bulge, 'theta': position_angle, 'disk radius': radius_disk, 'disk b/a': minor_major_ratio_disk}}

                        writestring = '202 [%s] %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %s\n' % (source_index, ra, dec, v_mag, v_minus_i, bulge_total_ratio, radius_bulge, minor_major_ratio_bulge, position_angle, radius_disk, minor_major_ratio_disk, position_angle, z)
                        pixfile.write(writestring)

                        source_index += 1
                        if source_index % (xpoints * ypoints) == 0:
                            file_index += 1
                            pixfile.close()
                            pixfile = open(path_to_file(file_index), 'a')

with open(pickle_file, 'wb') as f:
    pickle.dump(object_dict, f)
