"""Shared plotting style, palette and typography for the manuscript figures.

The case-level table is read from one path, defaulting to ../data/case_level_tables.xlsx
in the same way as the analysis scripts; override it with --data.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11.5,
    'axes.edgecolor': '#333333', 'axes.linewidth': 0.9,
    'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.7,
    'xtick.color': '#333333', 'ytick.color': '#333333',
    'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc', 'legend.fontsize': 9.5,
})

GREEN, BLUE, RED = '#48c8a0', '#3098d8', '#e84060'
NAVY, MID, LIGHT, LIGHTBLUE = '#203860', '#2058a0', '#70b0e0', '#b8d8f0'
ORANGE, PINK, SCAT = '#e89848', '#e87888', '#7f8fa6'

SERIF = {'family': 'DejaVu Serif', 'weight': 'bold'}
SANS = {'weight': 'bold'}


def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right'):
        if s not in keep:
            ax.spines[s].set_visible(False)


CASE = os.path.join(os.pardir, 'data', 'case_level_tables.xlsx')


def set_data(path):
    global CASE
    CASE = path


def add_data_arg(parser):
    parser.add_argument('--data', default=CASE, metavar='PATH',
                        help='case-level table (default: %(default)s)')
    return parser


def load():
    """One input table. The reference standard for the actual postoperative Cobb
    angle is the mean of the two blinded rater measurements, and every column the
    figures need is read from the same workbook."""
    B = pd.read_excel(CASE, 'T1_Patient_Raw_357').set_index('patient_id')
    return B, B, list(B.index)


def core(A, B, ids):
    post = B.loc[ids, 'post_cobb_main'].values.astype(float)
    ai = A.loc[ids, 'ai_cobb_main'].values.astype(float)
    pre = A.loc[ids, 'pre_cobb_main'].values.astype(float)
    corr = A.loc[ids, 'corr_rate_pct'].values.astype(float)
    return post, ai, pre, corr, ai - post


def stats_json():
    here = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(os.path.join(here, 'figstats.json')))
