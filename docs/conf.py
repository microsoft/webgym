# Configuration file for the Sphinx documentation builder.
# Mirrors the WebGym docs setup (Sphinx + MyST + Read the Docs theme).
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "AsyncWebRL"
copyright = "2026, AsyncWebRL Team"
author = "AsyncWebRL Team"
release = "v1"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",  # renders the math nodes MyST dollarmath emits
]

# MathJax 3 (tex-mml-chtml) does not bundle the ``textmacros`` extension, so
# escapes inside ``\text{...}`` (e.g. ``\_`` in ``\text{step\_adv}``) render as a
# literal backslash. Load it so ``\_ \$ \& \# \{ \}`` work in text mode.
mathjax3_config = {
    "loader": {"load": ["[tex]/textmacros"]},
    "tex": {"packages": {"[+]": ["textmacros"]}},
}

# MyST features used in the content (fenced ```{note}``` / ```{warning}```
# admonitions, ::: colon fences, etc.).
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "dollarmath",  # parse $...$ and $$...$$ as math (was missing -> rendered as literal text)
    "amsmath",     # support \begin{align} etc. inside math blocks
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
templates_path = ["_templates"]
# README.md is the docs-dir readme, not a page in the toctree.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "AsyncWebRL Documentation"
html_static_path = ["_static"]
# custom.css matches the WebGym look.
html_css_files = ["custom.css"]
