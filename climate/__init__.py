"""
CAMulator climate tools for CREDIT.

The scripts in this directory are run directly (``python climate/Quick_Climate.py``)
or imported as top-level modules with ``climate/`` on ``sys.path``; this file
imports nothing, so ``import climate`` and pytest collection of
``climate/camulator_gen2`` do not depend on them.

Main modules
------------
Quick_Climate : coupled CAMulator climate integration (gen1)
Model_State : model state and conservation fixers used by Quick_Climate
Make_Climate_Initial_Conditions : initial conditions from reanalysis data
Post_Process : spatial/temporal averaging and rescaling of climate output
WindPP : wind artifact filtering

Gen2 reproduction of the Hugging Face CAMulator: the separately installable
``camulator_gen2`` package in ``climate/camulator_gen2`` (see its README).
"""
