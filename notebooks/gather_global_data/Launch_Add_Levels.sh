#!/bin/bash
# PBS -N AddLevs
# PBS -A NAML0001
# PBS -l walltime=12:00:00
# PBS -o AddLevs.out
# PBS -e AddLevs.out
# PBS -q main
# PBS -l select=1:ncpus=5:mem=235GB
# PBS -m a
# PBS -M wchapman@ucar.edu

module load conda 
conda activate MILES

# List of years
years=(2006 2007)

# Loop through the years
for year in "${years[@]}"; do
  start_date="${year}-01-01"
  end_date="$((${year} + 1))-01-01"
  
  python GatherERA5_Add_levels.py --start_date ${start_date} --end_date ${end_date}
done
