.. MILES-CREDIT documentation master file, created by
   sphinx-quickstart on Wed Jul  3 11:39:28 2024.
   This file can be customized to suit your project, but it must
   contain the root `toctree` directive.

MILES-CREDIT Documentation
==========================

Welcome to the documentation for **MILES-CREDIT**, 
the **NSF NCAR Community Research Earth Digital Intelligent Twin** project. 
CREDIT is a machine learning-based research platform for understanding the best practices for training and operating global and regional AI autoregressive models, built as part of the NSF NCAR **Machine Integration and Learning for Earth Systems** (`MILES <https://ncar.github.io/miles>`_) group.

CREDIT enables users to train, run, and evaluate AI-based numerical weather and climate models. This documentation will guide you through installation, configuration, training, inference, evaluation, and extending the system with custom datasets and models.

**What you'll find here:**

- How to install CREDIT from source
- How to set up and train a model
- How to run inference and evaluate results
- How to contribute datasets, models, and enhancements
- Config file reference for reproducible HPC runs
- Tutorial videos for visual guidance

If you encounter issues or have suggestions, please open an issue on our GitHub repository. Contributions are welcome!

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Getting Started <getting-started.md>
   Installing CREDIT from source <installation.md>

.. toctree::
   :maxdepth: 2
   :caption: Configuration File

   Config Settings <config.md>

.. toctree::
   :maxdepth: 2
   :caption: Training and Inference

   Training a Model <Training.md>
   Running Inference <Inference.md>
   Evaluation and Metrics <Evaluation.md>
   Ensemble Training <Ensembles.md>
   Ensemble Inference <EnsemblesInference.md>
   Working with Loss Functions <Losses.md>

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   Contributing <contrib.rst>

.. toctree::
   :maxdepth: 1
   :caption: Adding New Models and Datasets (Coming Soon)

   Supported Model Architectures <Model_Architectures.md>
   Post Blocks <postblock.md>
   Dataset Structure <DataSets.md>

----

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

