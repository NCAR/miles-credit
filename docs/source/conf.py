from importlib.metadata import version as get_version
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "miles-credit"
copyright = "2024, University Corporation for Atmospheric Research"
author = "University Corporation for Atmospheric Research"
release = get_version("miles-credit")
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "autoapi.extension", "myst_parser"]
templates_path = ["_templates"]
exclude_patterns = []

# Render docstring "Attributes:" sections as :ivar: fields instead of standalone
# .. attribute:: directives. autoapi already documents each attribute from the
# class body, so the directive form produced "duplicate object description"
# warnings; :ivar: fields keep the descriptions without registering a second
# indexed object.
napoleon_use_ivar = True

myst_enable_extensions = ["colon_fence"]
# Generate slug anchors for h1-h3 headings so in-page links like
# [text](#heading-slug) resolve instead of raising myst.xref_missing.
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
autoapi_dirs = ["../../credit", "../../applications"]
# Drop "imported-members" from the defaults so package __init__ re-exports are
# not documented a second time under the package (they are already documented in
# the submodule that defines them). This removes the "duplicate object
# description" warnings.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
html_logo = "_static/credit_logo.png"

# Warnings that cannot be cleanly resolved in the source docstrings:
#   - ref.python: attribute type hints (padding/nlon/nlat) resolve to multiple
#     equally-named classes; disambiguation would require fully-qualified hints.
#   - autoapi.python_import_resolution: autoapi's static analyzer cannot follow
#     some runtime imports; harmless to the generated docs.
suppress_warnings = ["ref.python", "autoapi.python_import_resolution"]
