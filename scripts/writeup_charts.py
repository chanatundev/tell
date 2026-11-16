"""Scientific column charts using saved training/evaluation data only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures'
COLORS=['#354f6b','#36877f']


def build_charts():
    OUT.mkdir(parents=True,exist_ok=True)
    v1=json.loads((ROOT/'frozen/report.json').read_text())
    v2=json.loads((ROOT/'runs/v2/report.json').read_text())
    original=json.loads((ROOT/'results/heldout_v1/heldout_results.json').read_text())['tell']
    final=json.loads((ROOT/'results/post_unseal_v2/report.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelsize':7.5,'axes.titlesize':8.4,
                         'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})
    fig,ax=plt.subplots(figsize=(3.65,1.45))
    x=np.arange(3);w=.34
    for offset,run,color,label,hatch in [(-w/2,v1,COLORS[0],'V1',''),(w/2,v2,COLORS[1],'V2','//')]:
        y=[r['train_loss'] for r in run['history']]
        bars=ax.bar(x+offset,y,w,label=label,color=color,hatch=hatch,edgecolor='white',linewidth=.5,zorder=3)
        ax.bar_label(bars,labels=[f'{a:.3f}' for a in y],fontsize=6.5,padding=2)
    ax.set_xticks(x,['Epoch 1','Epoch 2','Epoch 3'])
    ax.set_ylim(0,.33);ax.set_yticks([0,.1,.2,.3])
    ax.set_ylabel('Training loss (BCE)')
    ax.set_title('A  Training loss over three epochs',loc='left',fontweight='bold',pad=8)
    ax.legend(frameon=False,ncols=2,loc='upper right',handlelength=1)
    ax.grid(axis='y',color='#e1e5e9',linewidth=.5,zorder=0)
    fig.subplots_adjust(left=.16,right=.98,top=.79,bottom=.18)
    p1=OUT/'tell-epoch-loss.png';fig.savefig(p1,dpi=450,facecolor='white');plt.close(fig)

    values=[[original['paired_acc'],original['recall_twins'][0]/50,original['false_alarm_reals'][0]/50],
            [final['paired_accuracy'],final['metrics']['recall'],final['metrics']['far']]]
    ci=[[original['paired_ci'],original['recall_twins'][2],original['false_alarm_reals'][2]],
        [final['paired_ci'],final['metrics']['recall_ci'],final['metrics']['far_ci']]]
    fig,ax=plt.subplots(figsize=(3.65,1.8))
    for i,(label,color,hatch) in enumerate([('V1: frozen before the test',COLORS[0],''),('V2: more training data',COLORS[1],'//')]):
        y=np.array(values[i])*100
        err=np.array([[y[j]-100*ci[i][j][0] for j in range(3)],
                      [100*ci[i][j][1]-y[j] for j in range(3)]])
        bars=ax.bar(x+(i-.5)*w,y,w,label=label,color=color,hatch=hatch,edgecolor='white',linewidth=.5,zorder=3)
        ax.errorbar(x+(i-.5)*w,y,yerr=err,fmt='none',ecolor='#253240',elinewidth=.7,capsize=2,zorder=4)
        for b,a in zip(bars,y):
            low = a < 20
            ax.text(b.get_x()+b.get_width()/2+(.11 if low else 0),a+2 if low else 4,
                    f'{a:.0f}%',ha='left' if low else 'center',va='bottom',
                    color='#253240' if low else 'white',fontsize=7,fontweight='bold',zorder=5)
    ax.set_xticks(x,['AI scored\nabove real','AI receipts\ncaught','Real receipts\nwrongly flagged'])
    ax.set_ylim(0,103);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Rate (%)')
    ax.set_title('B  AI receipt detection: V1 vs V2',loc='left',fontweight='bold',pad=25)
    ax.legend(frameon=False,ncols=2,loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0,handlelength=1)
    ax.grid(axis='y',color='#e1e5e9',linewidth=.5,zorder=0)
    fig.subplots_adjust(left=.16,right=.98,top=.73,bottom=.22)
    p2=OUT/'tell-v1-v2.png';fig.savefig(p2,dpi=450,facecolor='white');plt.close(fig)
    return p1,p2

if __name__=='__main__':
    print(*build_charts(),sep='\n')
