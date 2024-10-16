# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../conexionBBDD'))
# Cuando cree otra carpeta con archivos de python tengo que añadir otro enlace y Generar archivos .rst para todas las carpetas
#sys.path.insert(0, os.path.abspath('C:/Users/Gliglo/OneDrive - Universidad Politécnica de Madrid/Documentos/UPM/TFG/Proyecto_TFG/AFERNANDEZ_ms_2024/carpeta1'))

project = 'AFERNANDEZ_MS_2024'
copyright = '2024, Ángela Fernández'
author = 'Ángela Fernández'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
