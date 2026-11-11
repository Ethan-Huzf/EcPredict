"""Plot MAE versus actual training percentage or count from run metrics."""
from pathlib import Path
from collections import defaultdict
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True,help='Run directories, each with metrics.json')
    p.add_argument('--x',choices=['fraction','count'],default='fraction')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    groups=defaultdict(list)
    for path in args.runs:
        m=json.loads((path/'metrics.json').read_text())
        desc=f'DM (P = {m["permutations"]})' if m['descriptor']=='dm' else f'OPT3D (nsc = {m["nsc"]})'
        key=(desc,m['sampling'],m['target'])
        x=m['n_train'] if args.x=='count' else 100*m['n_train']/(m['n_train']+m['n_test'])
        groups[key].append((x,m['test_mae_meV']))
    targets={k[2] for k in groups}
    if len(targets)>1:raise ValueError('Plot signed-V and magnitude tasks separately')
    plt.rcParams.update({'font.family':'serif','font.size':11,'font.weight':'bold',
                         'axes.labelweight':'bold','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(6,4.3))
    colors=['#6F7781','#1F77B4','#D95F02','#16969B'];descs=sorted({k[0] for k in groups})
    for (desc,sampling,_),values in sorted(groups.items()):
        values=sorted(values);random=sampling=='random';color=colors[descs.index(desc)%len(colors)]
        ax.plot(*zip(*values),ls='--' if random else '-',marker='o',mfc='white' if random else color,
                color=color,lw=1.6,ms=4,label=f'{desc}, {sampling}')
    ax.set(xlabel='Training fraction (%)' if args.x=='fraction' else 'Training configurations',
           ylabel='Test MAE (meV)')
    for spine in ax.spines.values():spine.set_visible(True)
    ax.grid(axis='y',color='#DFE5EB',ls='--',lw=.6);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=600,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


if __name__=='__main__':main()
