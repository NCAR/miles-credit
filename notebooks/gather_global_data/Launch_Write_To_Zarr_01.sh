#!/bin/bash
# PBS -N LZ_01
# PBS -A NAML0001
# PBS -l walltime=12:00:00
# PBS -o LZ_01.out
# PBS -e LZ_01.out
# PBS -q main
# PBS -l select=1:ncpus=5:mem=230GB
# PBS -m a
# PBS -M wchapman@ucar.edu

module load conda 
conda activate MILES

# List of years
years=(1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991)

# Loop through the years
for year in "${years[@]}"; do
  python Zarr_Append_Levels.py $year 
done
