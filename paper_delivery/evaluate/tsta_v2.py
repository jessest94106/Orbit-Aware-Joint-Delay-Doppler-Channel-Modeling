"""Stationarity time T_sta vs elevation (Sec. V-C, eqs. 32-33) on the drawn-city geometry (physical_model.py, drawn PPP cities, wall image at antenna height,
exact edge point, analytic visibility, ground gated by LoS), unconditional weights, overhead pass, omni, 2 GHz.
Definition: F(u) = 1 - sum_i w_i exp(-(taudot_i u)^2/(2 T_s^2) - (nudot_i u)^2/(2 sig_nu^2)) / sum_i w_i,
T_sta = inf{u: F(u) > eps}, eps 0.05, T_s 33 ns, sig_nu = 1/(N T_s) = 0.1 Hz; rates by central differences (D = 0.5 s,
Doppler by dt = 50 ms). Paths pooled over NC cities (averaged density). Output data/model/tsta_v2_nNCITY.json; per-elevation cache in data/model/cache_tsta_nNCITY/.
Usage: python3 paper_delivery/evaluate/tsta_v2.py [NCITY]   (paper: 1000 cities per elevation, seeds 100+10*beta; ~1-2 h on 2 cores)"""
import sys, json, os, time, numpy as np
from multiprocessing import Pool
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PD.parent/'model')); OUT = PD/'data'/'model'
import physical_model as pm
TT=33.3333e-9; SN=0.1; EPS=0.05; DT=0.05; D=0.5
def collect(M,t0,city):
    ids=None; recs={}
    for u in (-D-DT,-D,-D+DT,-DT,0.,DT,D-DT,D,D+DT):
        p=M.paths(t0+u,city); recs[u]=p
        ids=p['id'] if ids is None else np.intersect1d(ids,p['id'],assume_unique=True)
    def tau_at(u):
        p=recs[u]; _,i1,_=np.intersect1d(ids,p['id'],assume_unique=True,return_indices=True); return p['tau'][np.searchsorted(p['id'],ids)] if np.all(np.diff(p['id'])>0) else p['tau'][[np.where(p['id']==k)[0][0] for k in ids]]
    def sel(u,key):
        p=recs[u]; idx=np.searchsorted(p['id'],ids); return p[key][idx]
    # physical_model returns ids sorted? ensure by sorting each record
    for u in recs:
        o=np.argsort(recs[u]['id']); recs[u]={k:(v[o] if isinstance(v,np.ndarray) and v.shape[:1]==recs[u]['id'].shape else v) for k,v in recs[u].items()}
    T0=sel(0.,'tau'); nu=lambda u: -pm.FC*(sel(u+DT,'tau')-sel(u-DT,'tau'))/(2*DT)
    N0=nu(0.); rt=(sel(D,'tau')-sel(-D,'tau'))/(2*D); rn=(nu(D)-nu(-D))/(2*D)
    p0=recs[0.]; mech=sel(0.,'mechanism'); amp=sel(0.,'amp'); prob=sel(0.,'prob'); wgt=sel(0.,'weight')
    w=amp*prob*wgt                       # unconditional: ground carries E[1_0] W_g, walls/edges their visibility probability
    return dict(w=w,rt=rt,rn=rn,mech=mech,tau=T0,nu=N0)
def job(j):
    scene,b,NC,seed=j; fn=OUT/f'cache_tsta_n{NC}'/f'{scene}_{b:g}.json'
    if os.path.exists(fn): return scene,b,json.load(open(fn))
    t_start=time.time(); M=pm.Model(scene); rng=np.random.default_rng(seed); t0=pm.reference_time(b)
    W=[];RT=[];RN=[];ME=[]
    for _ in range(NC):
        c=collect(M,t0,M.draw_city(rng)); W.append(c['w']);RT.append(c['rt']);RN.append(c['rn']);ME.append(c['mech'])
    W=np.concatenate(W);RT=np.concatenate(RT);RN=np.concatenate(RN);ME=np.concatenate(ME)
    def F(u,m=None):
        k=np.exp(-(RT*u)**2/(2*TT*TT)-(RN*u)**2/(2*SN*SN)); mm=np.ones(len(W),bool) if m is None else m
        return float(np.sum(W[mm]*(1-k[mm]))/W.sum())
    lo,hi=0.,3000.
    for _ in range(40):
        m=.5*(lo+hi)
        if F(m)>EPS: hi=m
        else: lo=m
    T=.5*(lo+hi)
    out=dict(T=T,frac={'g':W[ME==3].sum()/W.sum(),'w':W[ME==1].sum()/W.sum(),'e':W[ME==2].sum()/W.sum()},
             mig={'g':F(T,ME==3),'w':F(T,ME==1),'e':F(T,ME==2)},
             wall_nudot_mHzps=float(np.sum(W[ME==1]*np.abs(RN[ME==1]))/max(W[ME==1].sum(),1e-300)*1e3),
             wall_taudot_nsps=float(np.sum(W[ME==1]*np.abs(RT[ME==1]))/max(W[ME==1].sum(),1e-300)*1e9),
             ground_nudot_mHzps=float(np.abs(RN[ME==3]).mean()*1e3),ncity=NC,npaths=int(len(W)),secs=time.time()-t_start)
    out={k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in out.items()}
    json.dump(out,open(fn,'w')); print(scene,b,f"T={T:.1f}",{k:round(v,3) for k,v in out['frac'].items()},{k:round(v,4) for k,v in out['mig'].items()},f"wall nudot {out['wall_nudot_mHzps']:.1f} mHz/s  {out['secs']:.0f}s",flush=True)
    return scene,b,out
if __name__=='__main__':
    B=np.arange(10,86,5.); NC=int(sys.argv[1]) if len(sys.argv)>1 else 1000; (OUT/f'cache_tsta_n{NC}').mkdir(parents=True,exist_ok=True)
    jobs=[(s,float(b),NC,100+int(b*10)) for b in B for s in ('urban','rural')]
    with Pool(2) as p: res=p.map(job,jobs,chunksize=1)
    out={'beta':B.tolist(),'meta':dict(eps=EPS,T_s_ns=33.3,sig_nu_Hz=SN,ncity=NC,evaluator='physical_model v2, rural lambda_1D 0.02/m')}
    for s,b,o in res:
        out.setdefault(f'{s}_T',[]).append(o['T'])
        for k in ('g','w','e'): out.setdefault(f'{s}_frac_{k}',[]).append(o['frac'][k]); out.setdefault(f'{s}_mig_{k}',[]).append(o['mig'][k])
        out.setdefault(f'{s}_wall_nudot',[]).append(o['wall_nudot_mHzps']); out.setdefault(f'{s}_ground_nudot',[]).append(o['ground_nudot_mHzps'])
    json.dump(out,open(OUT/f'tsta_v2_n{NC}.json','w'),indent=1)
    for s in ('urban','rural'): print(s,'T',' '.join('%6.1f'%x for x in out[f'{s}_T']))
