# AIWQ-C Hindcasts

## Step I: Run `pregenerate_data_parallel.sh`

Pre-generates hindcast initial conditions and ensemble files (2000-01-01 to 2020-12-31) using a PBS job array.

```bash
mkdir -p logs logs_step1
# change the project key if needed...
qsub -J 0-766 pregenerate_data_parallel.sh
```
Also, check the instructions in pregenerate_data_parallel.sh

For testing, use a smaller job array range (e.g., 2 days):

```bash
qsub -J 0-1 pregenerate_data_parallel.sh
```

