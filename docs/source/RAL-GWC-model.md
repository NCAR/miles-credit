# Introduction

CONUS 404-based and FastEddy-based limited area models have been added to credit. The models are developed and supported by NCAR/RAL. 

-------------------------------------------------

## RAL GWC AI model updates

* The following scripts are added to `credit.datasets`
```
les_singlestep.py # single-step training  / inference datasets for FastEddy data
wrf_singlestep.py # single-step training / inference datasets for the CONUS 404 data
wrf_multistep.py # multi-step training dataset for the CONUS 404 data
```

* `credit.trainers`
```
trainerLES.py # single-step training routine
trainerWRF.py # single-step training routine
trainerWRF_multi.py # multi-step training routine
```

* `credit.losses`
```
les_loss.py # training loss class
```

* `credit.models`
```
swin_wrf.py # CONUS 404 model for prediction
dscale_wrf.py # CONUS 404 model for creating initialization
```

-------------------------------------------------

## Test new features

Go to the `ksha` branch, copy the following two folders to your directory, modify the directory information, and you can start testing new features.
```
/glade/work/ksha/DWC_runs/TEST_WRF/
/glade/work/ksha/DWC_runs/TEST_LES/
```

### Regional AI models

 * Single-step training using a single `Casper` GPU: 
```
credit_train_wrf -c /glade/work/ksha/DWC_runs/TEST_WRF/model_single_none.yml
```
 * Single-step training using FSDP on 4 `Derecho` GPUs:
```
qsub /glade/work/ksha/DWC_runs/TEST_WRF/launch_single.sh
```
* Multi-step training using FSDP on 4 `Derecho` GPUs:
```
qsub /glade/work/ksha/DWC_runs/TEST_WRF/launch_multi_01.sh
```
* Inference using a single `Casper` GPU:
```
credit_rollout_wrf -c /glade/work/ksha/DWC_runs/TEST_WRF/model_predict_none.yml
```
* Inference using FSDP on 4 `Derecho` GPUs:
```
qsub /glade/work/ksha/DWC_runs/TEST_WRF/launch_predict.sh 
```
### LES models

 * Single-step training using a single `Casper` GPU: 
```
credit_train_les -c /glade/work/ksha/DWC_runs/TEST_LES/model_single_none.yml
```

 * Single-step training using FSDP on 4 `Derecho` GPUs:
```
qsub /glade/work/ksha/DWC_runs/TEST_LES/launch_single.sh
```

* Inference using a single `Casper` GPU:
```
credit_rollout_les -c /glade/work/ksha/DWC_runs/TEST_LES/model_predict_none.yml
```

* Inference using FSDP on 4 `Derecho` GPUs:
```
qsub /glade/work/ksha/DWC_runs/TEST_LES/launch_predict.sh 
```
