# Project scope

You are a interactive data visualization expert: specifically with VTK and PyVista and general python. Your domain expertise in in Earth system modeling and analysis.

The overall goal is two-fold: 

1. to provide a robust interactive interface, with very low latency, for medium to high resolution earth-system model outputs (specifically AI / ML output from NCARs CREDIT platform)
2. and to then provide an interface to load some initial conditions for a pre-trained model, interactively perturb the initial conditions, launch a forecast, and then interactively analyize the differences between a perturbed and non-perturbed forecast

The dimensionality sizes I'm dealing with is (time=~100, variable=~6, vertical_level=~16, latitude=192, longitude=288) with float 32 values. However, the spatial domain may increase resolution eventually.

There are currently two scripts that provide a pathway to accomplish this:

1. demo_2D.py: 
    * This script works as is and handles data exploration with good latency with a demo file. This primarily handles use case #1 from above, and is more or less, fully functional as is.
2. demo_perturb.py
   * This is a skeleton script for use cased number 2 above. It pretty much just lays out a basic UI that seems like it will suffice. 

demo_2D.py currently has three "tabs": "single forecast", "compare forecasts", and "perturb and run". That script only utilizes "single forecast" and the others are placeholders. My thought was to break off the perturb, run and compare into a different script (demo_perturb.py) for simplicity.

## Overall objective:

I would like you to look at all of the code and see best how to proceed from a code strutural point of view first.

1. Should we keep the separate scipts for these very different types of analysis?
2. If so, there will be many utility functions, etc that would like be used / shared between both scripts -- if #1 is a good idea should we refactor to a more modular approach?

I would like to focus the development effort (not including potentiual modular refactor that will likely be first) on demo_perturb.py

General idea for demo_perturb.py
1. I would like to stick to the very similar 4-panel approach in demo_2D.py, but not include time-series or a vertical slice. The panels will likely be: forecast A, forecast B, Difference of A and B, and some other diagnostic.
2. They should all show the same variable and , potentially, contours. I would also like the same colorbar and cartopy map scheme as the other script.
3. I would like a placeholder function in place for "run CREDIT forecast" and for perturbing the data.
4. I'd like to demo this using the same data as demo_2D.py and just assume that forecast A and B are the same for this example (until I wire in more real data)

#### 

What did I miss? Please make sure to inspect the current state of the two scripts. Please ask clarifying questions, and run your plan by me before code changes.
