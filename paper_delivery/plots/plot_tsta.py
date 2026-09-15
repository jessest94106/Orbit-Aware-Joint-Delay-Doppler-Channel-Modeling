"""Fig. 16 (fig_tsta_0915): stationarity time vs elevation, urban and rural. Data: paper_delivery/data/model/tsta_v2_n1000.json (paper_delivery/evaluate/tsta_v2.py).
Output: paper_delivery/figures/fig_tsta_0915.pdf/.png"""
import json, numpy as np, matplotlib
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'; DR = PD/'data'/'references'; DS = PD/'data'/'scenemc'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
d=json.load(open(DM/'tsta_v2_n1000.json')); b=np.array(d['beta'])
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,
 'font.family':'serif','mathtext.fontset':'dejavuserif','axes.grid':True,'grid.alpha':0.35,'grid.linewidth':0.5})
BLUE=(0.0,0.447,0.698); ORANGE=(0.835,0.369,0.0)
fig,ax=plt.subplots(figsize=(3.5,2.2))
ax.semilogy(b,d['urban_T'],'-o',color=BLUE,ms=3.5,lw=1.4,label='Urban')
ax.semilogy(b,d['rural_T'],'--s',color=ORANGE,ms=3.5,lw=1.4,mfc='none',label='Rural')
ax.set_xlabel(r'Elevation $\beta_{\rm ele}$ (deg)'); ax.set_ylabel(r'$T_{\rm sta}$ (s)')
ax.set_xlim(8,87); ax.set_xticks([10,20,30,40,50,60,70,80]); ax.set_ylim(5,200)
tk=[5,10,20,50,100,200]; ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in tk])); ax.yaxis.set_minor_formatter(NullFormatter())
ax.legend(loc='upper right',framealpha=0.92,handlelength=2.2,labelspacing=0.3)
fig.tight_layout(pad=0.35); fig.savefig(FIG/'fig_tsta_0915.pdf'); fig.savefig(FIG/'fig_tsta_0915.png',dpi=220); print('ok')
