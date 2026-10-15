#!/usr/bin/env python3
"""Draw publication figures from retained or regenerated summary tables.

These figures use the working paper's numerical results with a reproducible
layout. They do not require the large simulation archives.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


def save(fig, output, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{name}.{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


def design(output, integrity):
    assert integrity['evaluation_markets'] == 16384
    assert integrity['reference_markets'] == 262144
    assert integrity['primary_intervals'] == 131072
    assert integrity['audit_markets'] == 18432
    fig, ax = plt.subplots(figsize=(10.4, 8.4))
    ax.set(xlim=(0, 10.4), ylim=(0, 8.4))
    ax.axis('off')

    def box(x, y, w, h, title, body):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                   linewidth=.8, facecolor='#f4f4f4', edgecolor='black'))
        ax.text(x + w / 2, y + h - .18, title, ha='center', va='top', fontsize=9.5,
                weight='bold')
        ax.text(x + w / 2, y + h - .51, body, ha='center', va='top', fontsize=8.5,
                linespacing=1.4)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                    linewidth=.9, color='black'))

    box(.15, 6.7, 3.0, 1.35, 'Train two procedures',
        'Independent training banks\nObservation and response functions\nCutoffs and response allowances')
    box(3.7, 6.7, 2.9, 1.35, 'Calibrate and validate',
        '12 calibration locations per bank\n512 markets per location\n36 preliminary validation settings')
    box(7.15, 6.7, 3.0, 1.35, 'Freeze the procedures',
        'Economic model and summaries\nFunctions, cutoffs, and allowances\nDomain and search budgets')
    arrow((3.24, 7.37), (3.59, 7.37))
    arrow((6.7, 7.37), (7.04, 7.37))
    box(.15, 4.85, 10., 1.15, 'Draw 256 parameter settings for the main experiment',
        '192 primary settings: 144 mixed + 16 belief drift + 16 funding drift + 16 allowance drift\n32 near-face settings and 32 near-corner settings; stress samples reported separately')
    arrow((8.65, 6.61), (8.65, 6.09))
    box(.15, 2.65, 4.7, 1.55, 'Construct response intervals',
        '64 evaluation markets per setting\n4 information designs × 2 fitted procedures\n131,072 intervals across all settings\n98,304 intervals in the primary sample')
    box(5.45, 2.65, 4.7, 1.55, 'Estimate independent reference means',
        '1,024 reference markets per parameter setting\nPaired transfer and control paths within each market\nMean month-19 response and its Monte Carlo error\nReference draws are independent of evaluation draws')
    arrow((2.5, 4.76), (2.5, 4.29))
    arrow((7.8, 4.76), (7.8, 4.29))
    box(.15, .45, 6.45, 1.55, 'Compare performance and information gains',
        'Coverage, empty intervals, and mean interval width\nPaired information contrasts on the same evaluation releases\n2,000 stratified bootstrap resamples of parameter settings\nUncertainty conditional on the two fitted procedures')
    box(7.15, .45, 3.0, 1.55, 'Audit fitted endpoints',
        'Expanded endpoint searches\n144 selected response endpoints\n128 fresh markets per endpoint\nNo recalibration after evaluation')
    arrow((2.5, 2.56), (2.5, 2.09))
    ax.plot([7.8, 7.8, 5.15], [2.56, 2.33, 2.33], color='black', linewidth=.9)
    arrow((5.15, 2.33), (5.15, 2.09))
    arrow((6.69, 1.22), (7.06, 1.22))
    ax.text(.15, .06, 'Main-run total: 296,960 simulated markets, including 18,432 endpoint-audit markets.',
            fontsize=8.5, ha='left')
    save(fig, output, 'figure_1_experiment')


def information(summary, output):
    panel = pd.read_csv(summary / 'panel_summary.csv')
    paired = pd.read_csv(summary / 'paired_contrasts.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), gridspec_kw={'width_ratios': [1, 1.08]})
    labels = ['O1\nHoldings', 'O2\n+ beliefs', 'O6\nComplete,\nlinked', 'O7\nComplete,\nunlinked']
    x = np.arange(4)
    for bank, shift, marker, face in [(0, -.10, 'o', 'black'), (1, .10, 's', 'white')]:
        rows = panel.query("group == 'primary' and metric == 'mean_width_pp' and bank == @bank")
        rows = rows.set_index('panel').loc[['O1', 'O2', 'O6', 'O7']]
        y = rows['estimate'].to_numpy()
        err = np.vstack([y - rows['ci95_low'].to_numpy(), rows['ci95_high'].to_numpy() - y])
        axes[0].errorbar(x + shift, y, yerr=err, fmt=marker, color='black', markerfacecolor=face,
                         markersize=5.5, capsize=3, linewidth=1, label=f'Procedure {bank + 1}')
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 2.15)
    axes[0].set_ylabel('Mean response-interval width (percentage points)')
    axes[0].set_title('A. Precision by information design', loc='left', fontweight='bold', pad=12)
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    comparisons = [('O1', 'O2'), ('O1', 'O6'), ('O7', 'O6')]
    for bank, shift, marker, face in [(0, -.10, 'o', 'black'), (1, .10, 's', 'white')]:
        base = paired.query("group == 'primary' and metric == 'width_reduction_percent' and bank == @bank")
        rows = base.set_index(['comparison', 'expanded']).loc[comparisons]
        y = rows['estimate'].to_numpy()
        err = np.vstack([y - rows['ci95_low'].to_numpy(), rows['ci95_high'].to_numpy() - y])
        axes[1].errorbar(np.arange(3) + shift, y, yerr=err, fmt=marker, color='black',
                         markerfacecolor=face, markersize=5.5, capsize=3, linewidth=1)
    axes[1].set_xticks(np.arange(3), ['O1 → O2\nBelief records', 'O1 → O6\nComplete linked\nrecords', 'O7 → O6\nRecord linkage'])
    axes[1].set_ylim(0, 58)
    axes[1].set_ylabel('Paired mean width reduction (%)')
    axes[1].set_title('B. Gains from additional information', loc='left', fontweight='bold', pad=12)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='.88', linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0, pad=8)
    fig.subplots_adjust(bottom=.26, wspace=.32, top=.9)
    fig.text(.08, .015, 'Primary sample: 12,288 evaluation markets per procedure–design combination. Whiskers: 95% Monte Carlo intervals\n'
             'from 2,000 stratified resamples of parameter settings. All primary response intervals contain the reference mean.',
             fontsize=8.2, va='bottom', linespacing=1.4)
    save(fig, output, 'figure_2_information')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.linewidth': .7})
    integrity = json.loads((args.summary / 'integrity.json').read_text())
    design(args.output, integrity)
    information(args.summary, args.output)
    print(f'Saved two figures as PNG, PDF, and SVG in {args.output}')


if __name__ == '__main__':
    main()
