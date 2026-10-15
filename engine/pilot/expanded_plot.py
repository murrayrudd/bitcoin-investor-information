"""Black-and-white publication figures from the saved validation summaries."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return list(csv.DictReader(open(p)))
def run(root):
    d=Path(root)/'summary';scope=read(d/'by_scope.csv');cases=read(d/'by_case.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250})
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.45),sharey=True)
    panels=['O1','O2','O6','O7'];x=np.arange(4)
    for ax,kind,title in zip(axes,['pure','mixed','boundary'],['Pure mechanisms','Mixed mechanisms','Boundary stress cases']):
        for bank,offset,color,hatch in [(0,-.18,'0.3',''),(1,.18,'white','///')]:
            y=[float(next(r for r in scope if r['bank']==str(bank) and r['kind']==kind and r['panel']==p)['mean_width_pp']) for p in panels]
            ax.bar(x+offset,y,.34,label=f'Bank {bank+1}',color=color,edgecolor='black',hatch=hatch,linewidth=.65)
        ax.set_xticks(x,panels);ax.set_title(title,fontsize=10);ax.grid(axis='y',color='.9',zorder=0);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean nonempty interval width\n(response percentage points)');axes[-1].legend(frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(d/'interval_widths.png');fig.savefig(d/'interval_widths.svg');plt.close(fig)
    names=[next(r['name'] for r in cases if int(r['truth_id'])==i) for i in range(12)]
    labels=['Pure belief','Pure funding','Pure limits','Previous mixed case','Reinforcing','Offsetting','Tightening limits','Declining','Reverse selection','High funding boundary','Low resources boundary','Opposed drifts boundary']
    fig,axes=plt.subplots(1,2,figsize=(9.5,5.1),sharey=True)
    for ax,bank in zip(axes,[0,1]):
        for p,offset,marker,color in [('O6',-.12,'o','black'),('O7',.12,'s','.55')]:
            selected=[next(r for r in cases if r['bank']==str(bank) and r['panel']==p and int(r['truth_id'])==i) for i in range(12)]
            y=np.array([float(r['coverage_pct']) for r in selected]);low=np.array([float(r['wilson_low_pct']) for r in selected]);high=np.array([float(r['wilson_high_pct']) for r in selected])
            ax.errorbar(y,np.arange(12)+offset,xerr=np.vstack([y-low,high-y]),fmt=marker,color=color,markersize=4,capsize=2,linewidth=.7,label=f'{p} '+('linked' if p=='O6' else 'unlinked'))
        ax.axvline(95,color='.4',linestyle=':',linewidth=.8);ax.set_xlim(0,101);ax.set_title(f'Independent bank {bank+1}',fontsize=10)
        ax.set_xlabel('Response coverage (%)');ax.set_yticks(np.arange(12),labels);ax.grid(axis='x',color='.92');ax.set_axisbelow(True)
    axes[0].invert_yaxis();axes[1].legend(frameon=False,fontsize=9,loc='lower left')
    fig.tight_layout();fig.savefig(d/'coverage_by_case.png');fig.savefig(d/'coverage_by_case.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
