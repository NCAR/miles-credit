# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "miles-credit"
copyright = "2024, University Corporation for Atmospheric Research"
author = "University Corporation for Atmospheric Research"
release = "2025.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "autoapi.extension", "myst_parser"]

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = ["colon_fence"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
autoapi_dirs = ["../../credit", "../../applications"]
html_logo = "_static/credit_logo.png"
