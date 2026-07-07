# CREDIT Gen 2 Overview
<div style="text-align: center;">
<img src="_static/credit_gen_2_small.png" width="90%" alt="CREDIT Gen 2 overview diagram" />
</div>

CREDIT Gen 2 addresses the need for an open, modular, composable framework to build ML Earth system prediction
models. CREDIT Gen 2 development has been guided by three key "signposts".

1. **Streamline user experience**: As a research tool, one of CREDIT's main goals is increasing speed to science.
We want users to be able to install the package and start running, training, and analyzing models within
minutes. We have added a CLI and restructured the config file to make interactions with CREDIT really straightforward.
2. **Reduce latency bottlenecks**: In our experience with CREDIT Gen 1, most of the compute time was being spent
loading data and performing pre- and post-processing operations. Therefore, we streamlined our datasets and converted
every pre-processing and post-processing step to PyTorch layers that can run on either CPU or GPU and are differentiable.
That means gradients can be calculated through physical transforms and time all the way to the initial conditions
in state space!
3. **Include more physics**: Even with the massive amount of data found in reanalyses and other model and observation 
datasets, many key processes and forcings are not explicitly represented, which results in the accumulation
of artifacts, oversmoothing, and improper coupling. We have made it much easier to include physics functions and
constraints at any point in the data and modeling pipeline. Not even the sky is the limit on what you could do with
this framework!

## Datasets



