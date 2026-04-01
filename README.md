# NSF NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT)

[![DOI](https://zenodo.org/badge/710968229.svg)](https://doi.org/10.5281/zenodo.14361005)

[PyPI](https://pypi.org/project/miles-credit/)

[CREDIT npj Climate and Atmospheric Science Article](nature.com/articles/s41612-025-01125-6)

## About
CREDIT is an open software platform to train and deploy AI atmospheric prediction models. CREDIT offers fast models
that can be flexibly configured both in terms of input data and neural network architecture. The interface is designed
to be user-friendly and enable fast spin-up and iteration. CREDIT is backed by the AI and atmospheric science expertise
of the MILES group and the NSF National Center for Atmospheric Research, leading to design choices that balance advanced
AI/ML with our physical knowledge of the atmosphere.

CREDIT has reached its first stable release with a full set of models, training, and deployment options. It continues
to be under active development. Please contact [the MILES group](mailto:milescore@ucar.edu) if you have any questions about CREDIT.

**New to CREDIT?** See [QUICKSTART.md](QUICKSTART.md) to get up and running locally, or the [full online Quickstart](https://miles-credit.readthedocs.io/en/latest/quickstart.html) for detailed guidance.

## AI assistant

`credit ask` is a unified AI assistant — it automatically runs in agent mode (reads your
files, runs shell commands, iterates until it has a confident answer) when an Anthropic key
is available, or falls back to simple chat (Groq/Gemini/OpenAI) otherwise.

```bash
pip install "miles-credit[ask]"

# Free — no card needed
export GROQ_API_KEY=gsk_...        # https://console.groq.com

# Or use your own Anthropic key for full agent mode (~$0.01–0.05/session)
export ANTHROPIC_API_KEY=sk-ant-...  # https://console.anthropic.com

credit ask "why is my loss not decreasing?"
credit ask -c my_run.yml "why did my training run crash?"
```

See the [AI Assistant documentation](https://miles-credit.readthedocs.io/en/latest/agent.html) for the full reference.

MILES CREDIT also provides more detailed [documentation](https://miles-credit.readthedocs.io/en/latest/) with installation
instructions, how to get started training and deploying models, how to interpret the config files, and full API docs.

## Citing CREDIT
If you are interested in using CREDIT as part of your research, please cite the following paper:
Schreck, J.S., Sha, Y., Chapman, W. et al. Community Research Earth Digital Intelligence Twin: a scalable framework 
for AI-driven Earth System Modeling. npj Clim Atmos Sci 8, 239 (2025). https://doi.org/10.1038/s41612-025-01125-6

# Model Weights and Data
Model weights for the CREDIT 6-hour WXFormer and FuXi models and the 1-hour WXFormer are available on huggingface.

* [6-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_6h)
* [1-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_1h)
* [6-Hour FuXi](https://huggingface.co/djgagne2/fuxi_6h)

Processed ERA5 Zarr Data are available for download through Globus (requires free account) through the [CREDIT ERA5 Zarr Files](https://app.globus.org/file-manager/collections/2fc90d8f-10b7-44e1-a6a5-cf844112822e/overview) collection.

Scaling/transform values for normalizing the data are available through Globus [here](https://app.globus.org/file-manager/collections/c5a23e21-1bee-4d1e-bb59-77c5dcee7c76). 

CREDIT also supports realtime runs generated from deterministic [Google Cloud GFS files](https://console.cloud.google.com/marketplace/product/noaa-public/gfs)
and raw cube sphere [GEFS files](https://console.cloud.google.com/marketplace/product/noaa-public/gfs-ensemble-forecast-system).

# Support
This software is based upon work supported by the NSF National Center for Atmospheric Research, a major facility sponsored by the 
U.S. National Science Foundation  under Cooperative Agreement No. 1852977 and managed by the University Corporation for Atmospheric Research. Any opinions, findings and conclusions or recommendations 
expressed in this material do not necessarily reflect the views of NSF. Additional support for development was provided by 
The NSF AI Institute for Research on Trustworthy AI for Weather, Climate, and Coastal Oceanography (AI2ES)  with grant
number RISE-2019758 and by Schmidt Sciences, LLC. 
