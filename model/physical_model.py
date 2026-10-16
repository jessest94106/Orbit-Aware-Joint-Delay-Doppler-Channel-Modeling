"""Per-path channel model of Sec. IV on one drawn city (the "v2" evaluator geometry).

Terminal at UE = (0, 0, 1.5 m); 550 km circular orbit, overhead pass by default (set_pass_offset to change);
carrier FC = 2 GHz; omni antennas.  Walls are a Poisson point process of finite segments with Rayleigh
heights (draw_city), plus the ground plane.  Model.paths(t, city) returns, at time t, every
  mechanism 1 : specular wall reflection  (image of the terminal in the wall plane, at the antenna height),
  mechanism 2 : roof-edge diffraction     (exact finite-distance stationary point on the roof edge, UTD coefficient),
  mechanism 3 : ground reflection         (image of the terminal in the ground plane),
  mechanism 4 : blocking-wall diffraction (NLoS only: over the roof edge of a wall that blocks the direct path, into
                its shadow; exact stationary point on the edge, single knife-edge loss J(v) of ITU-R P.526),
with relative power amp (direct path = 1), excess delay tau [s], analytic visibility probability prob
(the finite-segment blockage rule of Sec. IV / App. B; the ground path carries P_LoS), and, when
geometric_blockage=True, the ray-traced indicator visible against every other wall of the same city
(visible[-1] is the direct-path LoS event of that city).  Satellite-to-interaction segments are assumed clear.
Material kernels come from material.py.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc
from scipy.stats import qmc
sys.path.insert(0,str(Path(__file__).resolve().parent))
import material

C=299792458.;FC=2e9;UE=np.array([0.,0.,1.5]);RE=6371e3;RS=RE+550e3
OM=np.sqrt(3.986004418e14/RS**3)
OFF=0.0;ROT=np.eye(3)                     # overhead pass (beta_max = 90 deg) is the default used by every result in Sec. VI
def set_pass_offset(offset_m):
 """Rotate the circular orbit so that the ground track passes offset_m from the terminal (0 = overhead)."""
 global OFF,ROT
 OFF=offset_m/RE;ROT=np.array([[1.,0.,0.],[0.,np.cos(OFF),-np.sin(OFF)],[0.,np.sin(OFF),np.cos(OFF)]])
# scene table: (gamma [m] Rayleigh wall-height scale, lambda_b [m^-2] wall-centre density, l_f [m] wall length)
# urban : lambda_1D = 2 lambda_b l_f / pi = 0.025 /m (gamma 8 m, l_f 20 m)
# rural : lambda_1D = 0.02 /m (gamma 3 m, l_f 10 m), set so that P_LoS follows the TR 38.811 suburban/rural LOS row
SCENARIOS={'urban':(8.,.025*np.pi/40,20.),'rural':(3.,.02*np.pi/20,10.)}
def satellite(t):
 return ROT@(RS*np.array([np.sin(OM*t),0.,np.cos(OM*t)]))-np.array([0.,0.,RE])
def elevation(t):
 v=satellite(t)-UE;return np.arctan2(v[2],np.linalg.norm(v[:2]))
def reference_time(e):return brentq(lambda t:np.degrees(elevation(t))-e,-600,-1e-6)
def cross(a,b):return a[0]*b[1]-a[1]*b[0]

class Model:
 def __init__(self,scene,radius=2000.):
  self.scene=scene;self.gamma,self.density,self.width=SCENARIOS[scene];self.radius=radius
  self.lambda1d=2*self.density*self.width/np.pi
  self.eps=material.eps_concrete(FC)
 def tail(self,h):return self.gamma*np.sqrt(np.pi/2)*erfc(np.asarray(h)/(np.sqrt(2)*self.gamma))
 def visibility_probability(self,q):
  h=q[2];delta=h-UE[2]
  mean=np.divide(self.tail(UE[2])-self.tail(h),delta,out=np.full_like(h,np.exp(-UE[2]**2/(2*self.gamma**2))),where=abs(delta)>1e-8)
  return np.exp(-self.lambda1d*np.linalg.norm(q[:2],axis=0)*np.clip(mean,0,1))
 def draw_city(self,rng):
  n=rng.poisson(self.density*np.pi*self.radius**2)
  r=self.radius*np.sqrt(rng.random(n));a=rng.uniform(0,2*np.pi,n);p=rng.uniform(0,np.pi,n)
  return dict(x=r*np.cos(a),y=r*np.sin(a),angle=p,height=self.gamma*np.sqrt(-2*np.log(np.maximum(rng.random(n),1e-15))),weight=np.ones(n))
 def quadrature(self,power,seed):
  u=np.clip(qmc.Sobol(5,scramble=True,seed=seed).random_base2(power),1e-12,1-1e-12)
  sigs=[2.,15.,100.];mix=np.array([.2,.5,.25,.05]);comp=np.searchsorted(np.cumsum(mix),u[:,0]);r=np.empty(len(u))
  for j,sig in enumerate(sigs):
   mask=comp==j;tr=-np.expm1(-self.radius**2/(2*sig**2));r[mask]=sig*np.sqrt(-2*np.log1p(-u[mask,1]*tr))
  mask=comp==3;r[mask]=self.radius*np.sqrt(u[mask,1]);den=mix[3]*2*r/self.radius**2
  for j,sig in enumerate(sigs):den+=mix[j]*r/sig**2*np.exp(-r*r/(2*sig**2))/(-np.expm1(-self.radius**2/(2*sig**2)))
  a=2*np.pi*u[:,2]
  return dict(x=r*np.cos(a),y=r*np.sin(a),angle=np.pi*u[:,3],height=self.gamma*np.sqrt(-2*np.log(u[:,4])),weight=2*np.pi*self.density*r/(den*len(u)))
 def clear(self,end,city,owner=-1):
  """Finite UE-to-end ray; walls extend from terrain z=0 to sampled height."""
  ray=end-UE;e=np.array([np.cos(city['angle']),np.sin(city['angle'])]);centers=np.array([city['x'],city['y']])
  den=cross(ray[:2,None],e)
  t=np.divide(cross(centers,e),den,out=np.full(len(den),np.inf),where=abs(den)>1e-12)
  along=np.divide(cross(centers,ray[:2,None]),den,out=np.full(len(den),np.inf),where=abs(den)>1e-12)
  z=UE[2]+t*ray[2]
  hit=(t>1e-9)&(t<1-1e-9)&(abs(along)<=self.width/2)&(z>=0)&(city['height']>z)
  if owner>=0:hit[owner]=False
  return not np.any(hit)
 def paths(self,t,city,geometric_blockage=False):
  s=satellite(t);sg=s-UE;R=np.linalg.norm(sg);beta=elevation(t)
  x,y,p,H=city['x'],city['y'],city['angle'],city['height'];n=len(x)
  e=np.array([np.cos(p),np.sin(p),np.zeros(n)]);normal=np.array([-np.sin(p),np.cos(p),np.zeros(n)])
  d=normal[0]*x+normal[1]*y;ns=normal.T@s
  image=np.array([2*d*normal[0],2*d*normal[1],np.full(n,UE[2])])
  den=2*d-ns;z=np.divide(d-ns,den,out=np.full(n,np.nan),where=abs(den)>1e-12)
  # Interpolate from the nearby image UE to avoid subtracting orbital distances.
  one_minus_z=np.divide(d,den,out=np.full(n,np.nan),where=abs(den)>1e-12)
  q=image+one_minus_z*(s[:,None]-image);along=e[0]*(q[0]-x)+e[1]*(q[1]-y)
  reflection=(z>0)&(z<1)&(abs(along)<=self.width/2)&(q[2]>=0)&(q[2]<=H)
  ids=np.flatnonzero(reflection);qr=q[:,ids]
  ds=np.linalg.norm(qr-s[:,None],axis=0);dg=np.linalg.norm(qr-UE[:,None],axis=0)
  incidence=abs(np.sum(normal[:,ids]*(qr-s[:,None]),axis=0))/ds
  ar=material.fresnel_pavg(incidence,self.eps)*(R/(ds+dg))**2
  # Rationalized image-distance difference avoids cancellation at orbital ranges.
  v=image[:,ids]-UE[:,None];di=np.linalg.norm(s[:,None]-image[:,ids],axis=0)
  tr=(np.sum(v*v,axis=0)-2*np.sum(sg[:,None]*v,axis=0))/(di+R)/C

  # Exact finite-distance stationary point on each roof edge.
  eligible=(H>UE[2])&(z>0)&(z<1)&(q[2]>H)
  ii=np.flatnonzero(eligible);center=np.array([x[ii],y[ii],H[ii]]);ee=e[:,ii]
  def derivative(ell):
   point=center+ee*ell;v1=point-s[:,None];v2=point-UE[:,None]
   d1=np.linalg.norm(v1,axis=0);d2=np.linalg.norm(v2,axis=0)
   a1=np.sum(v1*ee,axis=0)/d1;a2=np.sum(v2*ee,axis=0)/d2
   return a1+a2,(1-a1*a1)/d1+(1-a2*a2)/d2
  fleft,_=derivative(-self.width/2);fright,_=derivative(self.width/2)
  valid=(fleft<=0)&(fright>=0);ii=ii[valid];center=center[:,valid];ee=ee[:,valid]
  a=center[0]*ee[0]+center[1]*ee[1];eta=(sg/R)@ee
  transverse=np.sqrt(np.maximum(np.sum((center-UE[:,None])**2,axis=0)-a*a,1e-15))
  ell=np.clip(eta*transverse/np.sqrt(1-eta*eta)-a,-self.width/2,self.width/2)
  for _ in range(5):
   f,fp=derivative(ell);ell=np.clip(ell-f/fp,-self.width/2,self.width/2)
  qe=center+ee*ell;d1=np.linalg.norm(qe-s[:,None],axis=0);d2=np.linalg.norm(qe-UE[:,None],axis=0)
  ui=(s[:,None]-qe)/d1;ud=(UE[:,None]-qe)/d2;eta=np.sum(ui*ee,axis=0)
  D2=material.utd_D2(ui,ud,ee[0],ee[1],normal[0,ii],normal[1,ii],d[ii],eta,d2*(1-eta**2),2*np.pi*FC/C,self.eps)
  ad=D2*R*R/(d1*d2*(d1+d2))
  v=qe-UE[:,None];te=((np.sum(v*v,axis=0)-2*np.sum(sg[:,None]*v,axis=0))/(d1+R)+d2)/C

  # Blocking-wall diffraction: walls that block the direct path (same test as clear) diffract over their roof edge into
  # the shadow, where mechanism 2 cannot reach (it needs the satellite and the terminal on the same side of the wall).
  # Stationary point as above; power J(v) of a single knife edge, v = 2 sqrt(excess path / lambda) (ITU-R P.526 4.1).
  den=cross(sg[:2,None],e[:2]);cen=np.array([x,y])
  tb=np.divide(cross(cen,e[:2]),den,out=np.full(n,np.inf),where=abs(den)>1e-12)
  ab=np.divide(cross(cen,sg[:2,None]),den,out=np.full(n,np.inf),where=abs(den)>1e-12);zb=UE[2]+tb*sg[2]
  jj=np.flatnonzero((tb>1e-9)&(tb<1-1e-9)&(abs(ab)<=self.width/2)&(zb>=0)&(H>zb))
  cb=np.array([x[jj],y[jj],H[jj]]);eb=e[:,jj]
  def dblock(ell):
   point=cb+eb*ell;v1=point-s[:,None];v2=point-UE[:,None]
   d1=np.linalg.norm(v1,axis=0);d2=np.linalg.norm(v2,axis=0)
   a1=np.sum(v1*eb,axis=0)/d1;a2=np.sum(v2*eb,axis=0)/d2
   return a1+a2,(1-a1*a1)/d1+(1-a2*a2)/d2
  fl,_=dblock(-self.width/2);fr,_=dblock(self.width/2)
  ok=(fl<=0)&(fr>=0);jj=jj[ok];cb=cb[:,ok];eb=eb[:,ok]                      # no stationary point on the edge: dropped, as for mechanism 2
  a=cb[0]*eb[0]+cb[1]*eb[1];eta=(sg/R)@eb
  transverse=np.sqrt(np.maximum(np.sum((cb-UE[:,None])**2,axis=0)-a*a,1e-15))
  lb=np.clip(eta*transverse/np.sqrt(1-eta*eta)-a,-self.width/2,self.width/2)
  for _ in range(5):
   f,fp=dblock(lb);lb=np.clip(lb-f/fp,-self.width/2,self.width/2)
  qb=cb+eb*lb;d1=np.linalg.norm(qb-s[:,None],axis=0);d2=np.linalg.norm(qb-UE[:,None],axis=0)
  v=qb-UE[:,None];tk=((np.sum(v*v,axis=0)-2*np.sum(sg[:,None]*v,axis=0))/(d1+R)+d2)/C
  vk=2*np.sqrt(np.maximum(tk*C,0)*FC/C)
  ak=10**(-(6.9+20*np.log10(np.sqrt((vk-.1)**2+1)+vk-.1))/10)*(R/(d1+d2))**2

  # Ground uses the image UE at z=-1.5 and the SAME LoS blockage event.
  gi=UE.copy();gi[2]=-UE[2];vg=gi-UE;rg=np.linalg.norm(s-gi)
  tg=(vg@vg-2*sg@vg)/(rg+R)/C
  ag=float(material.fresnel_pavg((s[2]+UE[2])/rg,3.5)*(R/rg)**2)
  pg=float(np.exp(-self.lambda1d*self.tail(UE[2])/np.tan(beta)))
  owner=np.r_[ids,ii,jj,-1];mechanism=np.r_[np.ones(len(ids),int),np.full(len(ii),2),np.full(len(jj),4),3]
  path_id=np.r_[2*ids,2*ii+1,2*n+1+jj,2*n]
  amp=np.r_[ar,ad,ak,ag];tau=np.r_[tr,te,tk,tg];prob=np.r_[self.visibility_probability(qr),self.visibility_probability(qe),self.visibility_probability(qb),pg]
  points=np.concatenate([qr,qe,qb,np.full((3,1),np.nan)],axis=1)
  visible=np.ones(len(amp),bool)
  if geometric_blockage:
   for j,k in enumerate(owner[:-1]):visible[j]=self.clear(points[:,j],city,int(k))
   visible[-1]=self.clear(s,city)
  weight=np.r_[city['weight'][ids],city['weight'][ii],city['weight'][jj],1.]
  assert np.all(np.isfinite(amp)) and np.all(amp>=0) and np.all(tau>=-1e-12)
  return dict(id=path_id,owner=owner,mechanism=mechanism,amp=amp,tau=tau,prob=prob,weight=weight,visible=visible,points=points,edge_stationarity_residual=derivative(ell)[0])

def moments(a,b,mode):
 _,ia,ib=np.intersect1d(a['id'],b['id'],assume_unique=True,return_indices=True)
 if mode=='geometry':joint=a['visible'][ia]*b['visible'][ib];pa=a['visible'];pb=b['visible']
 else:joint=np.minimum(a['prob'][ia],b['prob'][ib]);pa=a['prob'];pb=b['prob']
 z=np.sqrt(a['amp'][ia]*b['amp'][ib])*joint*a['weight'][ia]*np.exp(-2j*np.pi*FC*(b['tau'][ib]-a['tau'][ia]))
 return z.sum(),np.sum(a['amp']*pa*a['weight']),np.sum(b['amp']*pb*b['weight'])
