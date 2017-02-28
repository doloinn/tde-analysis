#!/bin/bash

# Exit on error
set -e

# Gibis directory location as input argument, default is .
GIBISDIR=${1:-.}

# Directory of my files
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Simulation parameters
PLACEHOLDER_INPUT_FILE=input_without_source.GibisInput
RUN_INPUT_FILE=input_with_source.GibisInput
GALAXY_DIR=galaxy_source_files
TDE_DIR=tde_source_files

# Change to Gibis directory (becomes pwd)
cd $GIBISDIR

echo "Galaxy simulations..."

# Iterates over galaxy pix files
for i in $(ls $MYDIR/$GALAXY_DIR)
do
    echo "$i:"
    echo "$i:" >> $MYDIR/galaxies.log
    # Replaces <sedreplacewithsourcefile> with location of pix files
    sed "s=<sedreplacewithsourcefile>=$MYDIR/$GALAXY_DIR/$i=g" $MYDIR/$PLACEHOLDER_INPUT_FILE > $MYDIR/$RUN_INPUT_FILE

    # Uncomment appropriate one of the following two lines (depending on setup). Simulation output is redirected to log file.
    # $PWD/RunGibis.sh gibis.GibisRun "davidlynn_$i" $MYDIR/$RUN_INPUT_FILE >> $MYDIR/galaxies.log 2>&1
    java -jar $PWD/dist/Gibis*.jar "davidlynn_$i" $MYDIR/$RUN_INPUT_FILE >> $MYDIR/galaxies.log 2>&1
done

echo "Galaxy simulations complete."
echo "TDE simulations..."

# Iterates over tde pix files
for i in $(ls $MYDIR/$TDE_DIR)
do
    echo "$i:"
    echo "$i:" >> $MYDIR/tdes.log
    # Replaces <sedreplacewithsourcefile> with location of pix files
    sed "s=<sedreplacewithsourcefile>=$MYDIR/$TDE_DIR/$i=g" $MYDIR/$PLACEHOLDER_INPUT_FILE > $MYDIR/$RUN_INPUT_FILE

    # Uncomment appropriate one of the following two lines (depending on setup). Simulation output is redirected to log file.
    # $PWD/RunGibis.sh gibis.GibisRun "davidlynn_$i" $MYDIR/$RUN_INPUT_FILE >> $MYDIR/galaxies.log 2>&1
    java -jar $PWD/dist/Gibis*.jar "davidlynn_$i" $MYDIR/$RUN_INPUT_FILE >> $MYDIR/tdes.log 2>&1
done

echo "TDE simulations complete."
echo "GIBIS simulations complete. Thank you."
