"""Publication-style plots from the saved pilot summary tables."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(output):
    out=Path(output);summary=out/'summary'
    with (summary/'by_scope.csv').open() as f:rows=list(csv.DictReader(f))
    index={(r['truth_kind'],r['case'],r['panel'],r['grid']):r for r in rows}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'normal','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(8.0,3.2),sharey=True)
    panels=['O1','O2','O6','O7'];x=np.arange(4)
    for ax,case,title in zip(axes,['clean_correct','selective_correct','selective_ignored'],['Random participation\nCorrect assumption','Selective participation\nCorrect assumption','Selective participation\nIgnored in fitting']):
        for kind,label,color,marker,offset in [('on_catalog','Anchor settings','black','o',-.07),('off_catalog','Between candidates','.45','s',.07)]:
            rr=[index[kind,case,p,'refined'] for p in panels]
            y=np.array([float(r['coverage'])*100 for r in rr]);lo=np.array([float(r['coverage_wilson_low'])*100 for r in rr]);hi=np.array([float(r['coverage_wilson_high'])*100 for r in rr])
            ax.errorbar(x+offset,y,yerr=[np.maximum(y-lo,0),np.maximum(hi-y,0)],label=label,color=color,marker=marker,ls='-' if kind=='on_catalog' else '--',lw=1,ms=4,capsize=2)
        ax.axhline(95,color='.5',ls=':',lw=1)
        ax.set_xticks(x,panels);ax.set_ylim(-3,104);ax.set_title(title,fontsize=9,pad=9)
        ax.set_yticks([0,25,50,75,95]);ax.grid(axis='y',color='.9',lw=.5)
    axes[0].set_ylabel('Intervals covering the oracle mean (%)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.01),ncol=2,frameon=False)
    fig.subplots_adjust(left=.075,right=.99,top=.81,bottom=.22,wspace=.14)
    for ext in ('png','svg'):fig.savefig(summary/f'coverage_by_panel.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.5,3.2),sharey=True)
    for ax,kind,title in zip(axes,['on_catalog','off_catalog'],['Anchor settings','Between candidate settings']):
        for grid,label,color,marker in [('primary','85 candidates','.5','s'),('refined','181 candidates','black','o')]:
            rr=[index[kind,'clean_correct',p,grid] for p in panels]
            ax.plot(x,[float(r['mean_width']) for r in rr],label=label,color=color,marker=marker,lw=1,ms=4)
        ax.set_xticks(x,panels);ax.set_title(title,fontsize=10);ax.grid(axis='y',color='.9',lw=.5)
    maximum=max(float(index[kind,'clean_correct',p,grid]['mean_width']) for kind in ('on_catalog','off_catalog') for p in panels for grid in ('primary','refined'))
    axes[0].set_ylim(0,maximum*1.1)
    axes[0].set_ylabel('Mean nonempty interval width\n(percentage points)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,-.01),ncol=2,frameon=False)
    fig.subplots_adjust(left=.10,right=.99,top=.87,bottom=.22,wspace=.25)
    for ext in ('png','svg'):fig.savefig(summary/f'width_and_refinement.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
