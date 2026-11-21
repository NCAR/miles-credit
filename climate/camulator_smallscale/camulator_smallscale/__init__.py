"""CAMulator fine-tuning experiments on small-scale variability, as CREDIT custom objects.

Builds on :mod:`camulator_gen2`, which reproduces the Hugging Face CAMulator training setup in the gen2 pipeline.
Every experiment arm is that setup plus one change (the loss, or noise layers), selected by config:

- :mod:`~camulator_smallscale.make_configs`: builds each arm's config from ``camulator_hf.yml``, for Derecho or for a
  local machine.

The AMSE, SDL + CRPS and wavelet pieces, the start-state perturbation preblock, the spectral metrics and the analysis
suite are added module by module.
"""

__version__ = "0.1.0"
