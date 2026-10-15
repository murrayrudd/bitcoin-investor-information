"""Publication-style static figures from the frozen validation summaries."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def run(root):
    root=Path(root); rows=read(root/'summary/by_case.csv'); dest=root/'summary'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    labels=['Pure beliefs','Pure funding','Pure position limits','Earlier mixed challenge',
            'Reinforcing changes','Offsetting changes','Tightening limits','Declining market',
            'Reverse participation','High funding boundary','Low resources boundary','Opposed drifts boundary',
            'New bulk 1','New bulk 2','New bulk 3','New bulk 4',
            'New face 1','New face 2','New face 3','New face 4',
            'New corner 1','New corner 2','New corner 3','New corner 4']
    fig, axes=plt.subplots(1, 2, figsize=(8.4, 7.2), sharey=True)
    y=np.arange(24)[::-1]
    for bank, ax in enumerate(axes):
        rr=sorted([r for r in rows if r['panel']=='O6' and int(r['bank'])==bank], key=lambda r:int(r['truth_id']))
        old=np.array([float(r['old_parameter_acceptance_pct']) for r in rr])
        new=np.array([float(r['true_parameter_acceptance_pct']) for r in rr])
        lo=np.array([float(r['parameter_acceptance_low_pct']) for r in rr])
        hi=np.array([float(r['parameter_acceptance_high_pct']) for r in rr])
        ax.hlines(y, old, new, color='.7', lw=.8, zorder=1)
        ax.plot(old, y, 'o', mfc='white', mec='.45', ms=4, label='Version 0.3')
        ax.errorbar(new, y, xerr=np.maximum(0., np.vstack([new-lo, hi-new])), fmt='o', color='black',
                    ms=3.8, lw=.7, capsize=1.8, label='Version 0.4 with 95% interval')
        ax.axvline(96.5, color='.4', ls=':', lw=1)
        ax.axhline(11.5, color='.6', lw=.7)
        ax.set_xlim(-2, 102); ax.set_xticks([0,25,50,75,100]); ax.set_title(f'Fitted bank {bank+1}', fontsize=11)
        ax.set_xlabel('True parameter retained (%)'); ax.grid(axis='x', color='.92', lw=.5)
    axes[0].set_yticks(y, labels)
    handles, legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.64, .005),
               ncol=2, fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=.27, bottom=.14, right=.985, top=.95, wspace=.13)
    for suffix in ('png','svg'):
        fig.savefig(dest/f'parameter_retention.{suffix}', dpi=250)
    plt.close(fig)

    pair=[]
    for bank in (0,1):
        for tid in range(24):
            a=next(r for r in rows if int(r['bank'])==bank and r['panel']=='O6' and int(r['truth_id'])==tid)
            b=next(r for r in rows if int(r['bank'])==bank and r['panel']=='O7' and int(r['truth_id'])==tid)
            pair.append((bank, tid, float(a['mean_width_pp']), float(b['mean_width_pp'])))
    fig, axes=plt.subplots(1, 2, figsize=(8.4, 4.), sharex=True, sharey=True)
    for bank, ax in enumerate(axes):
        for status, marker, color in [('previous', 'o', 'black'), ('new', '^', '.55')]:
            rr=[r for r in pair if r[0]==bank and ((r[1]<12)==(status=='previous'))]
            ax.scatter([r[3] for r in rr], [r[2] for r in rr], s=30, marker=marker,
                       facecolors='white', edgecolors=color, label='Earlier settings' if status=='previous' else 'New settings')
        end=max(max(r[2],r[3]) for r in pair)*1.06
        ax.plot([0,end],[0,end], color='.6', ls=':', lw=1)
        ax.set_xlim(0,end); ax.set_ylim(0,end); ax.set_aspect('equal')
        ax.set_xlabel('Unlinked mean interval width (pp)'); ax.set_title(f'Fitted bank {bank+1}', fontsize=11)
    axes[0].set_ylabel('Linked mean interval width (pp)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    for suffix in ('png','svg'):
        fig.savefig(dest/f'linked_interval_widths.{suffix}', dpi=250)
    plt.close(fig)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/repaired_pilot');a=p.parse_args();run(a.root)
