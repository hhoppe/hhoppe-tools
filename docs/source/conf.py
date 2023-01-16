"""Configuration file for the Sphinx documentation builder."""

# See https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hhoppe_tools'
copyright = '2023, Hugues Hoppe'
author = 'Hugues Hoppe'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',  # Recognize numpy and Google docstring styles.
    'sphinx.ext.autodoc',  # Support autofunction, autoclass, etc from docstrings.
    'sphinx.ext.viewcode',  # Add links to source code.
    'sphinx.ext.autosummary',  # Generate all autodoc directives automatically.
    # 'nbsphinx',             # Embed ipynb notebooks in the doc; requires "pip install nbsphinx".
    # 'sphinx.ext.githubpages',  # Publish HTML docs in GitHub Pages.
    # 'sphinx.ext.doctest',  # For running test snippets in the sphinx documentation.
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# nbsphinx_prompt_width = 0  # Omit notebook code cell prompts in html output.

autodoc_type_aliases = {
    '_NDArray': 'NDArray',
    '_DTypeLike': 'DTypeLike',
    '_ArrayLike': 'ArrayLike',
    '_Path': 'Path',
}

autodoc_member_order = 'alphabetical'  # Default 'alphabetical'; also 'bysource', 'groupwise'.

# Given meaning to single-backquotes `expression` within docstrings.
default_role = 'py:obj'  # Default is None; also 'py:obj', 'any', 'code'.

add_module_names = False  # Do not prepend objects with module name.  (Good for single module.)
python_use_unqualified_type_names = True  # Experimental; similar.

autosummary_generate = True  # Default is True; ['hhoppe_tools'] does not work.

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints/*']

# Big limitation which pdoc does not suffer from: long multiline function signatures.
# There is some recent ongoing work, e.g. https://github.com/sphinx-doc/sphinx/pull/11011

# Also: CSS trick in https://stackoverflow.com/questions/60146577 but too hacky.

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# https://www.sphinx-doc.org/en/master/usage/theming.html
html_theme = 'alabaster'
html_theme_options = {
    # 'rightsidebar': 'true',
    # 'relbarbgcolor': 'black',
    'body_min_width': '760px',
    # 'body_min_width': '1000px',
    'body_max_width': '1200px',
}
html_logo = 'logo.png'
html_favicon = 'favicon.ico'
html_static_path = ['_static']

# Themes define default as ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'].
# html_sidebars = {
#    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
# }
# Recommended and provided by alabaster theme:
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}


# html_css_files = ['custom.css']  # https://stackoverflow.com/a/64496917
# Add source/_static/custom.css; it appears in build/html/_static/custom.css
