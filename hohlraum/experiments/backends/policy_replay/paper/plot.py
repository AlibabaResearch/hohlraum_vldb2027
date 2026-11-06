"""Plot and summarize independently capacity-calibrated policy runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import sys
R=Path(sys.argv[1]).resolve()/'policy';OUT=Path(sys.argv[1]).resolve()/'figures';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(20261012,20261013,20261014);WORK=('agentx','playing');TH=(0,.1,.3,.5,.7,.9,1.);RATIOS=(0,.1,.25,.4,.5,.6,.8)
assert (R/'completed.json').exists()
def get(s,w,l,m='full_nocopy',t=.7,r=.5,v='full',family='policy'):
 d=R/f'seed{s}'/w/f'coverage{l}';ps=list(d.glob(f'{family}-{m.replace(":","-")}-t{t:.2f}-r{r:g}-{v}-c*.json'));assert len(ps)==1,(s,w,l,m,t,r,v,ps);return json.loads(ps[0].read_text())
def allrow(s,w,l):return get(s,w,l,'all',0,0,family='baseline')
def xy(row,ref):return row['written_blocks']/ref['written_blocks'],100*row['global_hits']/row['refs']
def meanxy(w,l,m='full_nocopy',t=.7,r=.5,v='full',family='policy'):return np.mean([xy(get(s,w,l,m,t,r,v,family),allrow(s,w,l)) for s in SEEDS],axis=0)
def larc(s,w,l):
 c=json.loads((R/f'seed{s}'/w/'capacities.json').read_text())[str(l)]['capacity_blocks']
 return get(s,w,l,f'larc:{round(c*256)}',0,0,family='baseline')
def envelope(pts):
 out=[];hi=-1
 for x,y in sorted(pts, key=lambda p:(float(p[0]),float(p[1]))):
  if y>hi+1e-7:out.append((x,y));hi=y
 return np.array(out)
def style(ax):
 ax.grid(color='#e2e5e8',lw=.45);ax.set_axisbelow(True);ax.tick_params(length=2,pad=2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6,'pdf.fonttype':42,'axes.linewidth':.6})
frontstats={}
for w in WORK:
 fig,ax=plt.subplots(figsize=(1.7,1.48));style(ax);visible_y=[]
 for label,kind,color,marker in [('Hohlraum','full_nocopy','#ba5037','o'),('Hohlraum-Admission','lru_nocopy','#9068af','s'),('Access-count','access','#838383','^'),('LARC','larc','#258977','D')]:
  pts=[]
  if kind in ('full_nocopy','lru_nocopy'):
   pts=[meanxy(w,80,kind,t,.5 if kind=='full_nocopy' else 0) for t in TH]
  else:
   for n in ((1,2,3,4,5,8,12,20) if kind=='access' else (.1,.2,.4,.8,1.6,3.2,6.4,16,64,256)):
    vals=[]
    for s in SEEDS:
     c=json.loads((R/f'seed{s}'/w/'capacities.json').read_text())['80']['capacity_blocks'];mode=f'{kind}:{n if kind=="access" else max(1,round(c*n))}'
     vals.append(xy(get(s,w,80,mode,0,0,family='baseline'),allrow(s,w,80)))
    pts.append(np.mean(vals,axis=0))
  points=envelope(pts);ax.plot(points[:,0],points[:,1],color=color,marker=marker,ms=3,lw=1.2,label=label)
  visible_y.extend(points[points[:,0]>=.2,1].tolist())
  if points[0,0]<.2<points[-1,0]:visible_y.append(float(np.interp(.2,points[:,0],points[:,1])))
 star=np.mean([xy(allrow(s,w,80),allrow(s,w,80)) for s in SEEDS],axis=0);ax.scatter(*star,marker='*',s=40,c='#333',label='Write-all')
 ax.set_title('AgentX' if w=='agentx' else 'Playing');ax.set_xlabel('SSD writes / LRU write-all');ax.set_ylabel('Global hit rate (%)');ax.set_xlim(.2,1.03)
 visible_y.append(float(star[1]));lo=min(visible_y);hi=max(visible_y);pad=max(.15,.04*(hi-lo));ax.set_ylim(lo-pad,min(100,hi+pad));ax.set_xticks([.2,.4,.6,.8,1.]);ax.yaxis.set_major_locator(MaxNLocator(4))
 ax.legend(loc='best',framealpha=.9,labelspacing=.15,handlelength=1.,fontsize=4.6)
 fig.subplots_adjust(left=.23,right=.98,bottom=.25,top=.85)
 for ext in ('pdf','png'):fig.savefig(OUT/f'policy-{w}-coverage-capacity-20261031.{ext}',dpi=220)
 plt.close(fig)
 savings=[]
 for s in SEEDS:
  rows={m:[get(s,w,80,m,t,.5 if m=='full_nocopy' else 0) for t in TH] for m in ('full_nocopy','lru_nocopy')}
  best={m:min((r for r in rs if r['global_hits']>=.99*r['theory_hits']),key=lambda r:r['written_blocks'],default=None) for m,rs in rows.items()}
  if all(best.values()):savings.append(100*(1-best['full_nocopy']['written_blocks']/best['lru_nocopy']['written_blocks']))
 frontstats[w]={'paired_write_savings_at_99pct_theory':savings,'mean':float(np.mean(savings)) if savings else None}
# Scope and ranking, and lifecycle signal ablations.
fig,axs=plt.subplots(1,2,figsize=(3.35,2.45));abstats={}
left=[('Hohlraum','full_nocopy','full','#c46d55','o'),('LRU ranking','lru_nocopy','full','#5c9d8f','^'),('All KV','full_copy','full','#939393','D'),('All KV + LRU','lru_copy','full','#4f7197','s')]
right=[('Hohlraum','full_nocopy','full','#c46d55','o'),('No session-return model','full_nocopy','no_return','#4f7197','s'),('No branch-reuse model','full_nocopy','no_branch','#5c9d8f','^')]
for panel,(ax,variants) in enumerate(zip(axs,(left,right))):
 style(ax);allpts=[]
 for label,m,v,color,marker in variants:
  pts=[]
  for s in SEEDS:
   row=get(s,'playing',80,m,.7,.5,v);ref=allrow(s,'playing',80) if panel==0 else get(s,'playing',80)
   pts.append(xy(row,ref) if panel==0 else (100*(row['written_blocks']/ref['written_blocks']-1),100*(row['global_hits']-ref['global_hits'])/row['refs']))
  pts=np.array(pts);av=pts.mean(axis=0);allpts.extend(pts.tolist());abstats[f'{panel}:{m}:{v}']=pts.tolist()
  ax.scatter(pts[:,0],pts[:,1],s=9,color=color,marker=marker,alpha=.55,lw=0);ax.scatter(*av,s=28,color=color,marker=marker,edgecolors='#333',lw=.45,label=label)
 p=np.array(allpts);lo=p.min(axis=0);hi=p.max(axis=0);delta=np.maximum(hi-lo,[.05,.15] if panel==0 else [1,.01]);ax.set_xlim(lo[0]-.15*delta[0],hi[0]+.15*delta[0]);ax.set_ylim(lo[1]-.17*delta[1],hi[1]+.17*delta[1])
 ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4))
 ax.set_xlabel('SSD writes / write-all' if panel==0 else 'Extra SSD writes (%)',fontsize=6.5);ax.set_ylabel('Global hit rate (%)' if panel==0 else 'Hit-rate change (pp)',fontsize=6.5)
 ax.set_title('(a) Scope and ranking' if panel==0 else '(b) Lifecycle signals',fontsize=7)
 ax.legend(loc='upper left',bbox_to_anchor=(-.13,-.30),frameon=False,fontsize=5.8,handletextpad=.2)
 if panel:ax.axhline(0,c='#999',lw=.5,ls='--');ax.axvline(0,c='#999',lw=.5,ls='--')
fig.subplots_adjust(left=.13,right=.985,top=.9,bottom=.4,wspace=.55)
for ext in ('pdf','png'):fig.savefig(OUT/f'policy-ablation-coverage-capacity-20261031.{ext}',dpi=240)
plt.close(fig)
# Same hit-write sensitivity form, with percentage allocations.
fig,ax=plt.subplots(figsize=(3.35,2.65));style(ax)
colors={'agentx':'#1769a0','playing':'#d65b35'};styles={80:'-',90:'--',99:':'};shapes={80:'o',90:'s',99:'^'}
sensitivity={}
for w in WORK:
 for l in (80,90,99):
  pts=np.array([meanxy(w,l,'lru_nocopy' if r==0 else 'full_nocopy',.7,r) for r in RATIOS]);sensitivity[f'{w}:c{l}']=pts.tolist()
  ax.plot(pts[:,0],pts[:,1],c=colors[w],ls=styles[l],lw=1.2)
  for r,(x,y) in zip(RATIOS,pts):ax.scatter(x,y,marker=shapes[l],s=9+24*r,facecolor='white' if r==0 else colors[w],edgecolor=colors[w],lw=.5)
  offsets={'agentx':{80:(3,-10),90:(2,7),99:(5,-10)},'playing':{80:(4,-9),90:(4,6),99:(12,-12)}}
  ax.annotate(f'c{l}',xy=pts[0],xytext=offsets[w][l],textcoords='offset points',fontsize=5.7,color=colors[w],annotation_clip=False)
  q=np.mean([xy(larc(s,w,l),allrow(s,w,l)) for s in SEEDS],axis=0)
  ax.scatter(*q,marker='D',s=15,facecolor='white',edgecolor=colors[w],lw=.7,zorder=4)
  ours=pts[RATIOS.index(.5)]
  ax.plot([q[0],ours[0]],[q[1],ours[1]],c=colors[w],ls=styles[l],lw=.8,alpha=.23,zorder=.7)
  ref=np.mean([xy(allrow(s,w,l),allrow(s,w,l)) for s in SEEDS],axis=0)
  ax.scatter(*ref,marker='*',s=32,c=colors[w],edgecolors='#333',lw=.35,zorder=4)
ax.set_xlabel('Normalized logical SSD writes');ax.set_ylabel('Global hit rate (%)');ax.set_xlim(0,1.04)
handles=[Line2D([],[],c=colors[w],label='AgentX' if w=='agentx' else 'Playing') for w in WORK]+[Line2D([],[],c='#555',ls=styles[l],marker=shapes[l],ms=3,label=f'c{l}') for l in (80,90,99)]
handles += [Line2D([],[],ls='',marker='D',mfc='white',mec='#555',ms=4,label='LARC'),Line2D([],[],ls='',marker='*',c='#555',ms=6,label='Write-all')]
leg=ax.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.20),frameon=False,fontsize=6,columnspacing=.7,handlelength=1.3);ax.add_artist(leg)
handles=[Line2D([],[],ls='',marker='o',ms=np.sqrt(9+24*r),mfc='white' if r==0 else '#888',mec='#888',label=f'{r*100:g}%') for r in (0,.25,.5,.8)]
ax.legend(handles=handles,title='Staging / total DRAM',title_fontsize=6,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.40),frameon=False,fontsize=6,columnspacing=.7,handletextpad=.3)
fig.subplots_adjust(left=.17,right=.98,top=.96,bottom=.36)
for ext in ('pdf','png'):fig.savefig(OUT/f'policy-sensitivity-coverage-capacity-20261031.{ext}',dpi=240)
plt.close(fig)
def compare(a,b):
 vals=[]
 for s in SEEDS:
  x=get(s,'playing',80,a);y=get(s,'playing',80,b)
  vals.append((100*(1-x['written_blocks']/y['written_blocks']),100*(x['global_hits']-y['global_hits'])/x['refs']))
 return np.mean(vals,axis=0).tolist()
summary=dict(frontiers=frontstats,ablation_points=abstats,scope=compare('full_nocopy','full_copy'),ranking_nocopy=compare('full_nocopy','lru_nocopy'),ranking_all=compare('full_copy','lru_copy'),sensitivity=sensitivity)
(R/'analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('sensitivity','ablation_points')},indent=2))
