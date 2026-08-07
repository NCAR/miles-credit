# Postblocks
Postblocks perform data transforms and diagnostic calculations after the model
has performed its forward pass. For WXFormer and current models, the first
step is to run the Reconstruct postblock to re-create the state dictionary.
Next, a bridgescaler transform block performs an inverse transform to return
data to physical instead of normalized values. Then, some fields that have been
log or square root transformed may need to have their inverse operations run.

Once all needed variables are back in physical ranges of values, diagnostic 
postblocks can be run to calculate derived fields like geopotential, perform
interpolation to pressure levels, enforce conservation laws, or run storm
tracking algorithms. If the algorithms are implemented in differentiable PyTorch,
the outputs could potentially be included in the loss function to enable physics
guidance during training.

## Summary of Available Postblocks