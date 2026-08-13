.. MILES-CREDIT documentation master file, created by
   sphinx-quickstart on Wed Jul  3 11:39:28 2024.
   This file can be customized to suit your project, but it must
   contain the root `toctree` directive.

CREDIT Documentation
==========================

Welcome to the documentation for **CREDIT**,
the **NSF NCAR Community Research Earth Digital Intelligence Twin** project.
CREDIT is an open foundational research platform for building machine learning Earth system prediction emulators.
It is developed and maintained primarily by the NSF NCAR **Machine Integration and Learning for Earth Systems**
(`MILES <https://ncar.github.io/miles>`_) group along with significant contributions from other NSF NCAR scientists
and engineers, interns, visitors, and collaborators across the world.

CREDIT enables users to train, run, and evaluate AI-based numerical weather and climate models. This documentation
will guide you through installation, configuration, training, inference, evaluation, and extending the system with
custom datasets and models. CREDIT's new Generation 2 restructuring and a more intuitive CLI make it easier than ever
to train your own emulator.

**New here?** Begin with `Get Started <quickstart.html>`_ — it gets you from zero to a running training job in under 10 minutes.

**What you'll find here:**

- How to install CREDIT
- How to set up and train a model
- How to run inference and evaluate results
- How to contribute datasets, models, and enhancements

If you encounter problems or have suggestions, please open an issue on our GitHub repository. :doc:`Contributions <contrib>` are welcome!

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   Get Started <quickstart.md>
   CLI <cli.md>


.. toctree::
   :maxdepth: 2
   :caption: Generation 2 Components

   Overview <gen2_overview.md>
   Datasets <Datasets.md>
   Preblocks <Preblocks.md>
   Postblocks <postblocks_gen2.md>
   Models <Models_gen2.md>
   Loss Functions <Losses.md>
   Verification Metrics <Metrics.md>
   Custom Objects <Custom.md>


.. toctree::
   :maxdepth: 2
   :caption: Training and Inference

   Training a Model <Training.md>
   Monitoring with TensorBoard <tensorboard.md>
   Running Inference <Inference.md>
   Forecast API Server <serve.md>
   AI Agent <agent.md>
   Evaluation and Metrics <Evaluation.md>
   Ensemble Training <Ensembles.md>
   Ensemble Inference <EnsemblesInference.md>


.. toctree::
   :maxdepth: 2
   :caption: Contributing

   Contributing <contrib.md>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   Overview <api/index>

.. toctree::
   :maxdepth: 1
   :caption: Generation 1

   Installing CREDIT from source <installation.md>
   Config Settings <config.md>
   Prepare New Dataset <prepare_new_dataset.md>
   Supported Model Architectures <Model_Architectures.md>
   Post Blocks <postblock.md>
   Losses (Gen 1) <Losses_gen1.md>
   Training (Gen 1) <Training_gen1.md>
   Data Pipeline for Downscaling <downscaling-pipeline.md>
   RAL GWC regional model <RAL-GWC-model.md>


----

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

