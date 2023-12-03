# This file is part of the Metron AI ArDaGen (https://github.com/OndrejSzekely/metron_ai_ardagen).
# Copyright (c) 2023 Ondrej Szekely.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3. This program
# is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details. You should have received a copy of the GNU General Public
# License along with this program. If not, see <http://www.gnu.org/licenses/>.

# SKIP PRE-COMMIT HOOKS
#   pylint: skip-file
#   type: ignore
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../."))


# -- Project information -----------------------------------------------------

project = "Metron AI - DeepForge"
copyright = "2023, ondra.szekely@gmail.com"
author = "ondra.szekely@gmail.com"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Using MyST UI to support all MyST features.
    "myst_parser",  # For parsing MyST markdown files.
    "sphinxemoji.sphinxemoji",  # Sphinx extension to render emojis.
    "sphinx.ext.githubpages",  # For proper rendering of the docs on GitHub pages.
    "sphinx_design",
    "sphinx.ext.autosummary",
    "sphinx_togglebutton",  # For collapsible code blocks.
    "sphinx_copybutton",  # For copy button on code blocks.
    "sphinxcontrib.mermaid",
    "sphinx_subfigure",  # https://sphinx-subfigure.readthedocs.io/en/latest/
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- autodoc configuration ---------------------------------------------------
autodoc_mock_imports = ["typeguard", "numpy", "hydra", "omegaconf"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": True,
    "exclude-members": "__weakref__ , __module__ , __dict__",
}

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = ["dollarmath", "colon_fence"]

# -- sphinx-subfigure configuration ------------------------------------------------------
numfig = False
