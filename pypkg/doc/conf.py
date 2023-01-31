# -*- coding: utf-8 -*-

import os
from pathlib import Path
from partis.utils.sphinx import basic_conf

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# configuration
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

src_dir = Path(__file__).parent
# autodoc_capture = src_dir.parent.parent / 'build' / 'autodoc'
autodoc_capture = None

globals().update( basic_conf(
  package = 'p4est',
  author = "p4est",
  email = "p4est@ins.uni-bonn.de",
  copyright_year = '2023',
  src_dir = src_dir,
  autodoc_capture = autodoc_capture ) )

html_logo = os.fspath(src_dir / '_static' / 'logo2.png')


html_theme_options = {
  #-----------------------------------------------------------------------------
  # light theme
  'light_css_variables': {

    'color-foreground-primary' : '#31363B',
    'color-foreground-muted' : '#454545',
    'color-foreground-secondary' : '#292727',
    'color-foreground-border' : '#BAB9B8',

    'color-background-primary' : '#EFF0F1',
    'color-background-secondary' : '#d6e6de',
    'color-background-border' : '#b1b5b9',
    'color-inline-code-background' : '#e7f9f0',

    'color-brand-primary' : '#39845f',
    'color-brand-content' : '#39845f',

    'color-highlighted-background' : '#3daee90',

    'color-guilabel-background' : '#30506b80',
    'color-guilabel-border' : '#1c466a80',

    'color-highlight-on-target' : '#e2d0b7',
    'color-problematic' : '#875e05'
  },
  #-----------------------------------------------------------------------------
  # dark theme
  'dark_css_variables': {

    'color-foreground-primary' : '#eff0f1',
    'color-foreground-muted' : '#736f6f',
    'color-foreground-secondary' : '#a7aaad',
    'color-foreground-border' : '#76797c',

    'color-background-primary' : '#373b39',
    'color-background-secondary' : '#4f5955',
    'color-background-border' : '#51575d',
    'color-inline-code-background' : '#424745',

    'color-brand-primary' : '#6af4af',
    'color-brand-content' : '#6af4af',

    'color-highlighted-background' : '#3daee90',

    'color-guilabel-background' : '#30506b80',
    'color-guilabel-border' : '#1c466a80',

    'color-highlight-on-target' : '#7c5418',
    'color-problematic' : '#e6c07b'
  } }
