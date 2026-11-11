"""Draw a closed-frame parity plot from run_experiment.py prediction CSVs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='SVG path')
    p.add_argument('--target', choices=['signed','abs'], required=True)
    p.add_argument('--colorbar', action='store_true')
    args = p.parse_args()
    train, test = [pd.read_csv(args.run/f'{s}_predictions.csv') for s in ['train','test']]
    for df in [train,test]:
        if not np.isfinite(df[['DFT_target_meV','predicted_meV']]).all().all():
            raise ValueError('Prediction CSV contains nonfinite values')
        expected = abs(df.V_DFT_meV) if args.target == 'abs' else df.V_DFT_meV
        if not np.allclose(df.DFT_target_meV, expected):
            raise ValueError('Requested target does not match CSV reference labels')
    values = np.concatenate([d[['DFT_target_meV','predicted_meV']].to_numpy().ravel()
                             for d in [train,test]])
    lo = min(-10, np.floor(values.min()/10)*10-5)
    hi = max(10, np.ceil(values.max()/10)*10+5)
    cmap = LinearSegmentedColormap.from_list('warm_density',
           ['#B88A00','#F28E1C','#D62728','#A20F5B','#111111'], N=1024)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
                         'font.weight':'bold','axes.labelweight':'bold','font.size':12,
                         'mathtext.fontset':'stix','svg.fonttype':'none'})
    fig,ax = plt.subplots(figsize=(5.3 if args.colorbar else 5,5))
    if args.colorbar:
        fig.subplots_adjust(left=.16,right=.84,bottom=.14,top=.98)
    else:
        fig.subplots_adjust(left=.17,right=.98,bottom=.14,top=.98)
    ax.scatter(train.DFT_target_meV,train.predicted_meV,s=4,c='#2474A6',alpha=.28,lw=0)
    x,y = test.DFT_target_meV.to_numpy(),test.predicted_meV.to_numpy()
    edges = np.linspace(lo,hi,59)
    hist,_,_ = np.histogram2d(x,y,bins=(edges,edges))
    ix,iy = [np.clip(np.searchsorted(edges,v,side='right')-1,0,57) for v in [x,y]]
    density = np.log1p(hist[ix,iy]);span=np.ptp(density)
    density = (density-density.min())/span if span else np.zeros_like(density)
    order=np.argsort(density)
    ax.scatter(x[order],y[order],c=density[order],cmap=cmap,vmin=0,vmax=1,
               s=10,marker='x',linewidths=.5,alpha=.8)
    ax.plot([lo,hi],[lo,hi],'--',color='#555555',lw=.9)
    symbol=r'$|V|$' if args.target=='abs' else r'$V$'
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=f'DFT {symbol} (meV)',ylabel=f'Predicted {symbol} (meV)')
    ax.set_aspect('equal');ax.grid(False)
    for s in ax.spines.values():s.set_visible(True);s.set_color('#555555')
    ax.legend(handles=[Line2D([],[],ls='',marker='o',color='#2474A6',alpha=.4,markersize=3,label='Training set'),
                       Line2D([],[],ls='',marker='x',color='#D58B0B',markersize=4,label='Test set')],
              frameon=False,loc='upper left',fontsize=10)
    mae=float(np.mean(abs(x-y)))
    ax.text(.975,.035,f'MAE = {mae:.3f} meV',transform=ax.transAxes,ha='right',fontsize=10,
            bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='#555555',lw=.8))
    if args.colorbar:
        cax=fig.add_axes([.88,.23,.024,.60])
        cax.imshow(np.linspace(0,1,1024)[:,None],cmap=cmap,vmin=0,vmax=1,
                   origin='lower',aspect='auto',interpolation='bilinear')
        cax.set_xticks([]);cax.set_yticks([])
        cax.text(.5,1.025,'High',transform=cax.transAxes,ha='center',fontsize=9)
        cax.text(.5,-.055,'Low',transform=cax.transAxes,ha='center',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=600,bbox_inches='tight',pad_inches=.035,facecolor='white')
    fig.savefig(args.output.with_suffix('.png'),dpi=600,bbox_inches='tight',pad_inches=.035,facecolor='white')
    plt.close(fig)
    print(f'Actual test MAE: {mae:.6f} meV; exported {args.output}')


if __name__=='__main__':main()
