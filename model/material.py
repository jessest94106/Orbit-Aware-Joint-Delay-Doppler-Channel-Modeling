"""Material and diffraction kernels shared by the channel model.

eps_concrete : ITU-R P.2040 concrete permittivity at frequency f (Hz).
fresnel_pavg : polarization-averaged Fresnel power reflection coefficient |Gamma|^2
               for cos(theta_i) and relative permittivity eps (Sec. IV-B, specular walls and ground).
utd_F, utd_fresnel, utd_D2 : UTD wedge diffraction of a finitely conducting right-angle wedge
               (n = 3/2; ITU-R P.526 Sec. 4.5, Kouyoumjian-Pathak), used for roof-edge diffraction (Sec. IV-C).
This file is the material kernel imported by physical_model.py; it holds no scene or orbit parameters.
"""
import numpy as np

E0=8.854e-12
def eps_concrete(f):
    fg=f/1e9; return 5.24-1j*0.0462*fg**0.7822/(2*np.pi*f*E0)

def fresnel_pavg(cos_t, eps):
    ct = np.clip(cos_t, 1e-9, 1.0)
    root = np.sqrt(eps - (1-ct**2) + 0j)
    gte = (ct-root)/(ct+root)
    gtm = (eps*ct-root)/(eps*ct+root)
    return 0.5*(np.abs(gte)**2 + np.abs(gtm)**2)

# ---- UTD wedge coefficient of a finitely conducting wedge (ITU-R P.526 sec. 4.5; Kouyoumjian-Pathak 1974) ----
_utd_t=np.linspace(0,np.sqrt(12.0),60000)
_utd_c=np.concatenate([[0j],np.cumsum(0.5*(np.exp(-1j*_utd_t[1:]**2)+np.exp(-1j*_utd_t[:-1]**2))*np.diff(_utd_t))])
def utd_F(x):
    """transition function F(x)=2j sqrt(x) e^{jx} int_{sqrt x}^inf e^{-jt^2} dt (table x<=12, asymptotic above)"""
    x=np.maximum(np.asarray(x,float),1e-12); sx=np.sqrt(np.minimum(x,12.0))
    tail=np.sqrt(np.pi)/2*np.exp(-1j*np.pi/4)-(np.interp(sx,_utd_t,_utd_c.real)+1j*np.interp(sx,_utd_t,_utd_c.imag))
    F=2j*np.sqrt(x)*np.exp(1j*x)*tail
    big=x>12.0
    return np.where(big,1+1j/(2*x)-3/(4*x**2),F)
def utd_fresnel(cos_t,eps):
    ct=np.clip(cos_t,1e-9,1.0); root=np.sqrt(eps-(1-ct**2)+0j)
    return (ct-root)/(ct+root),(eps*ct-root)/(eps*ct+root)      # TE, TM
def utd_D2(vs,vg,ex,ey,nx,ny,dpl,eta,L,k0,eps,n=1.5):
    """|D|^2, polarization-averaged. vs, vg: unit directions (3,N) FROM the edge point toward the
    satellite and toward the GT; (ex,ey) edge direction; (nx,ny) facade normal, dpl=n.c;
    eta = Rhat.e (Keller cone: sin beta0 = sqrt(1-eta^2)); L = distance parameter."""
    sgn=-np.sign(dpl); ox,oy=sgn*nx,sgn*ny                    # outward normal, toward the GT side
    def ang(v):                                                # angle from the wall face (down), through the exterior
        a=np.arctan2(v[0]*ox+v[1]*oy,-v[2]); return np.where(a<0,a+2*np.pi,a)
    p1,p2=ang(vs),ang(vg)
    sb0=np.sqrt(np.maximum(1-eta**2,1e-6))
    def cot(x): return np.cos(x)/np.where(np.abs(np.sin(x))<1e-12,1e-12,np.sin(x))
    def apm(x,sign):
        N=np.round((x+sign*np.pi)/(2*n*np.pi)); return 2*np.cos((2*n*np.pi*N-x)/2)**2
    dm,sp=p2-p1,p2+p1
    T1=cot((np.pi+dm)/(2*n))*utd_F(k0*L*apm(dm,+1))
    T2=cot((np.pi-dm)/(2*n))*utd_F(k0*L*apm(dm,-1))
    T3=cot((np.pi-sp)/(2*n))*utd_F(k0*L*apm(sp,-1))
    T4=cot((np.pi+sp)/(2*n))*utd_F(k0*L*apm(sp,+1))
    c0=np.abs(vs[0]*nx+vs[1]*ny)                               # wall incidence: cos(theta_i)
    cn=np.abs(vs[2])                                           # roof incidence: cos of the arrival angle
    g0te,g0tm=utd_fresnel(c0,eps); gnte,gntm=utd_fresnel(cn,eps)
    pref=1/(2*n*np.sqrt(2*np.pi*k0)*sb0)
    Dte=pref*(T1+T2+g0te*T3+gnte*T4); Dtm=pref*(T1+T2+g0tm*T3+gntm*T4)
    return 0.5*(np.abs(Dte)**2+np.abs(Dtm)**2)
