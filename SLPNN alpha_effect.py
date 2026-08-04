"""
============================================================================
  SLPNN Alpha Comparison Study
  α ∈ {0.50, 0.75, 0.90}  
============================================================================
"""

import numpy as np
from scipy.special import gamma as Γ
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import warnings, time
from mpmath import mp, nsum, inf as mpinf, gamma as mpg, power, mpf

warnings.filterwarnings('ignore'); mp.dps = 16
matplotlib.rcParams.update({
    'font.family':'DejaVu Serif','font.size':11,
    'axes.labelsize':12,'axes.titlesize':12,'axes.titleweight':'bold',
    'figure.dpi':150,'axes.grid':True,'grid.alpha':0.22,
    'lines.linewidth':2.0,'legend.fontsize':9,
    'xtick.labelsize':10,'ytick.labelsize':10,
})
SEP = '='*72

# ═══════════════════════════════════════════════════════════════════════
#  1  PHYSICAL PARAMETERS  
# ═══════════════════════════════════════════════════════════════════════
A_HAT=0.05; W_S=0.30; EPS_S=0.10; B_REF=0.80; C_STAR=1.00; T_END=3.0
P  = W_S  / (1.0-A_HAT)
Q  = EPS_S/ (1.0-A_HAT)**2
EPS_BC = EPS_S/(1.0-A_HAT)
ROUSE  = P/Q

def C_INF(Z): return np.exp(-ROUSE*np.asarray(Z,float))

ALPHA_LIST   = [0.50, 0.75, 0.90]
ALPHA_COLORS = {0.50:'#1565C0', 0.75:'#2E7D32', 0.90:'#B71C1C'}
ALPHA_FILL   = {0.50:'#BBDEFB', 0.75:'#C8E6C9', 0.90:'#FFCDD2'}
ALPHA_LS     = {0.50:'-',       0.75:'--',      0.90:'-.'}

print(SEP)
print('  SLPNN Alpha Comparison: α ∈ {0.50, 0.75, 0.90}')
print(SEP)
print(f'  P={P:.5f}  Q={Q:.5f}  EPS_BC={EPS_BC:.5f}  Rouse={ROUSE:.4f}')

# ═══════════════════════════════════════════════════════════════════════
#  2  SHIFTED LEGENDRE BASIS  
# ═══════════════════════════════════════════════════════════════════════
def SLP(N, x):
    x=np.atleast_1d(np.asarray(x,float))
    P=np.zeros((N+1,len(x))); P[0]=1.
    if N>=1: P[1]=2*x-1
    for n in range(1,N):
        P[n+1]=((2*n+1)*(2*x-1)*P[n]-n*P[n-1])/(n+1)
    return P

def dSLP(N,x):
    x=np.atleast_1d(np.asarray(x,float))
    Pv=SLP(N,x); D=np.zeros((N+1,len(x)))
    if N>=1: D[1]=2.
    for n in range(1,N): D[n+1]=2*(2*n+1)*Pv[n]+D[n-1]
    return D

def d2SLP(N,x):
    x=np.atleast_1d(np.asarray(x,float))
    Dv=dSLP(N,x); D2=np.zeros((N+1,len(x)))
    for n in range(1,N): D2[n+1]=2*(2*n+1)*Dv[n]+D2[n-1]
    return D2

def _slp_coeffs(n):
    from math import comb as C
    return [(k,(-1)**(n-k)*C(n,k)*C(n+k,k)) for k in range(n+1)]

def caputo_SLP(N, alpha, tau):
    tau=np.atleast_1d(np.asarray(tau,float))
    res=np.zeros((N+1,len(tau)))
    for n in range(N+1):
        for(k,c) in _slp_coeffs(n):
            if k>=1: res[n]+=c*(Γ(k+1)/Γ(k+1-alpha))*tau**(k-alpha)
    return res

# ═══════════════════════════════════════════════════════════════════════
#  3  SEMI-ANALYTICAL EIGENVALUES  
# ═══════════════════════════════════════════════════════════════════════
N_SA=60; _m=-P/(2*Q); _b=abs(_m); _k=(B_REF-W_S)/EPS_BC+_b

def _tr(xi):
    if abs(xi)<1e-9: return _k+_b
    return (_k+_b)*xi*np.cos(xi)-(xi**2-_k*_b)*np.sin(xi)

print('\n  Computing SA eigenvalues (α-independent) …', end=' ', flush=True)
_xs=np.linspace(0.05,N_SA*np.pi+1,500000); _fv=np.array([_tr(x) for x in _xs])
_xsa=[]
for i in range(len(_fv)-1):
    if _fv[i]*_fv[i+1]<0:
        try:
            r=brentq(_tr,_xs[i],_xs[i+1],xtol=1e-13)
            if r>0.05 and(not _xsa or abs(r-_xsa[-1])>0.05): _xsa.append(r)
        except: pass
    if len(_xsa)>=N_SA: break
XI_SA=np.array(_xsa[:N_SA]); LAM_SA=Q*(XI_SA**2+_m**2)

def _phi(n,Z):
    xi=XI_SA[n]; a=xi/_k
    return np.exp(_m*np.asarray(Z,float))*(np.sin(xi*Z)+a*np.cos(xi*Z))
def _Inn(n):
    xi=XI_SA[n]; a=xi/_k
    return(1+a**2)/2+(a**2-1)*np.sin(2*xi)/(4*xi)+a*(1-np.cos(2*xi))/(2*xi)
def _Cn(n):
    xi=XI_SA[n]; a=xi/_k; b=_b; d2=xi**2+b**2
    Ish=(b*np.cosh(b)*np.sin(xi)-xi*np.sinh(b)*np.cos(xi))/d2
    Ich=(xi*np.sinh(b)*np.sin(xi)+b*np.cosh(b)*np.cos(xi)-b)/d2
    return 2*(Ish+a*Ich)/_Inn(n)
_CN=np.array([_Cn(n) for n in range(N_SA)])
print(f'done  ({N_SA} modes,  λ₁={LAM_SA[0]:.4f})')

# ═══════════════════════════════════════════════════════════════════════
#  4  NETWORK PARAMETERS  
# ═══════════════════════════════════════════════════════════════════════
N_Z=15; N_T=15; N_W=N_Z*N_T; W_BC=100.0; sw_bc=np.sqrt(W_BC)
N_RUNS=5; T_BC_EVAL_MIN=0.20

def cgn(n,a=0.,b=1.):
    k=np.arange(1,n+1)
    return 0.5*(1+np.cos(np.pi*(2*k-1)/(2*n)))*(b-a)+a

N_PDE_Z,N_PDE_T,N_BC=30,30,35
Z_pde=cgn(N_PDE_Z); T_pde=cgn(N_PDE_T,0.02,1.); T_bc=cgn(N_BC,0.05,1.)


P_pde_Z  = SLP(N_Z-1, Z_pde)
D1_pde_Z = dSLP(N_Z-1, Z_pde)
D2_pde_Z = d2SLP(N_Z-1, Z_pde)
P_bc_T   = SLP(N_T,    T_bc)[1:,:]
P_zero_j = np.array([(-1.)**j for j in range(1,N_T+1)])
ETA_bc   = P_bc_T - P_zero_j[:,None]

nn=np.arange(N_Z,dtype=float)
P_z1=np.ones(N_Z); D1_z1=nn*(nn+1)
P_z0=(-1.)**nn;    D1_z0=(-1.)**(nn-1)*nn*(nn+1)
BC1_Z = EPS_BC*D1_z1 + W_S*P_z1
BC2_Z = EPS_BC*D1_z0 + (W_S-B_REF)*P_z0

J_bc1_base = sw_bc*np.einsum('i,jq->qij',BC1_Z,ETA_bc).reshape(N_BC,N_W)
J_bc2_base = sw_bc*np.einsum('i,jq->qij',BC2_Z,ETA_bc).reshape(N_BC,N_W)
b_bc       = sw_bc*(-W_S)*np.ones(N_BC)


NZ_CMP,NT_CMP=200,300
Z_cmp=np.linspace(0.,1.,NZ_CMP)
T_cmp=np.linspace(0.05,T_END,NT_CMP)

# ═══════════════════════════════════════════════════════════════════════
#  5  PER-ALPHA COMPUTATION LOOP
# ═══════════════════════════════════════════════════════════════════════
RESULTS = {}   

for ALPHA in ALPHA_LIST:
    print(f'\n{SEP}')
    print(f'  α = {ALPHA}')
    print(SEP)

   
    print(f'  Building ML LUT (α={ALPHA}) …', end=' ', flush=True)
    _t0=time.time()
    def _ml_raw(xf, a=ALPHA):
        if xf<=1e-9: return 1.
        if xf>80:
            g1,g2,g3=Γ(1-a),Γ(1-2*a),Γ(1-3*a)
            return 1/(g1*xf)-1/(g2*xf**2)+1/(g3*xf**3)
        return float(nsum(lambda k:power(mpf(-xf),k)/mpg(a*k+1.),
                          [0,mpinf],tol=1e-13,error=False))
    _xl=np.unique(np.concatenate([
        np.linspace(1e-4,0.15,50),np.linspace(0.15,2.5,100),
        np.exp(np.linspace(np.log(2.5),np.log(250.),350))]))
    _yl=np.array([_ml_raw(x,ALPHA) for x in _xl])
    _MLC=CubicSpline(_xl,_yl)
    def ML(x_in, a=ALPHA, cs=_MLC):
        x=np.atleast_1d(np.asarray(x_in,float)); out=np.ones_like(x)
        big=x>250.; out[big]=1./(Γ(1-a)*x[big])
        mid=(~big)&(x>0); out[mid]=np.clip(cs(x[mid]),0.,1.)
        return float(out[0]) if np.ndim(x_in)==0 else out
    print(f'done ({time.time()-_t0:.1f}s)')

    
    def SA(Z_arr, t_arr, a=ALPHA, ml=ML):
        Z=np.asarray(Z_arr,float); t=np.asarray(t_arr,float)
        out=C_INF(Z)[:,None]*np.ones((1,len(t)))
        PHI=np.column_stack([_phi(n,Z) for n in range(N_SA)])
        ta=np.maximum(t,1e-8)**a
        for n in range(N_SA):
            out+=(_CN[n]*ml(LAM_SA[n]*ta))[None,:]*PHI[:,n:n+1]
        return out

    
    P_pde_T  = SLP(N_T,T_pde)[1:,:]
    ETA_pde  = P_pde_T - P_zero_j[:,None]
    CAP_T    = caputo_SLP(N_T,ALPHA,T_pde)[1:,:]/(T_END**ALPHA)

    T1=np.einsum('ip,jq->pqij',P_pde_Z, CAP_T  ).reshape(N_PDE_Z*N_PDE_T,N_W)
    T2=np.einsum('ip,jq->pqij',D1_pde_Z,ETA_pde).reshape(N_PDE_Z*N_PDE_T,N_W)
    T3=np.einsum('ip,jq->pqij',D2_pde_Z,ETA_pde).reshape(N_PDE_Z*N_PDE_T,N_W)
    J_pde=T1-P*T2-Q*T3; b_pde=np.zeros(N_PDE_Z*N_PDE_T)

    J    =np.vstack([J_pde,J_bc1_base,J_bc2_base])
    b_rhs=np.concatenate([b_pde,b_bc,b_bc])

    
    D_col=np.maximum(np.linalg.norm(J,axis=0),1e-14)
    D_inv=1./D_col; J_sc=J*D_inv[None,:]
    kap  =D_col.max()/D_col.min()
    print(f'  Column norms: [{D_col.min():.2e}, {D_col.max():.2e}]  κ ~ {kap:.1e}')

  
    print(f'  SVD …', end=' ', flush=True)
    _sv=time.time()
    U_sc,s_sc,Vt_sc=np.linalg.svd(J_sc,full_matrices=False)
    s_thr=s_sc.max()*max(970,N_W)*np.finfo(float).eps*100
    s_eff=np.where(s_sc>s_thr,s_sc,0.)
    n_rank=int((s_eff>0).sum())
    _Utr=U_sc.T@b_rhs
    _w_sc_qr=Vt_sc.T@np.where(s_eff>0,_Utr/s_sc,0.)
    E_qr=float(np.dot(J_sc@_w_sc_qr-b_rhs,J_sc@_w_sc_qr-b_rhs))
    print(f'done ({time.time()-_sv:.1f}s)  rank={n_rank}/{N_W}  '
          f'σ∈[{s_sc[s_eff>0].min():.3e},{s_sc.max():.3e}]  E_qr={E_qr:.4e}')

    
    def marquardt(seed):
        rng=np.random.default_rng(seed)
        r_mask=s_eff>0
        w_sc=Vt_sc[r_mask].T@rng.normal(0.,.01,int(r_mask.sum()))
        lam=float(s_sc.max()**2/100.); hist=[]
        for _ in range(1000):
            r=J_sc@w_sc-b_rhs; E=float(r@r); hist.append(E)
            if E<1e-10: break
            dw=Vt_sc.T@(s_eff/(s_eff**2+lam)*(U_sc.T@r))
            w_new=w_sc-dw
            E_new=float(np.dot(J_sc@w_new-b_rhs,J_sc@w_new-b_rhs))
            if E_new<E: w_sc=w_new; lam=max(lam/4.,1e-16)
            else:        lam=min(lam*2.,1e12)
        w_sc=Vt_sc[r_mask].T@(Vt_sc[r_mask]@w_sc)
        return w_sc*D_inv, hist

    print(f'  Marquardt ({N_RUNS} seeds) …')
    all_hists=[]; best_E=np.inf; best_h=None
    for sd in range(N_RUNS):
        w_opt,hist=marquardt(sd); fE=hist[-1]
        tag='✓' if fE/E_qr<1.02 else f'ratio={fE/E_qr:.2f}'
        print(f'    Seed {sd}: E={fE:.4e}  {tag}')
        all_hists.append((sd,hist,fE))
        if fE<best_E: best_E=fE; best_h=hist

   
    W_SLPNN=(_w_sc_qr*D_inv).reshape(N_Z,N_T)

  
    def SLPNN(Z_arr,t_arr,W=W_SLPNN):
        Z_arr=np.asarray(Z_arr,float); t_arr=np.asarray(t_arr,float)
        tau=t_arr/T_END
        PZ=SLP(N_Z-1,Z_arr)
        PT=SLP(N_T,tau)[1:,:]
        ETA=PT-P_zero_j[:,None]
        return 1.+PZ.T@W@ETA

    def SLPNN_dZ(Z_arr,t_arr,W=W_SLPNN):
        Z_arr=np.asarray(Z_arr,float); t_arr=np.asarray(t_arr,float)
        tau=t_arr/T_END
        DZ=dSLP(N_Z-1,Z_arr)
        PT=SLP(N_T,tau)[1:,:]
        ETA=PT-P_zero_j[:,None]
        return DZ.T@W@ETA

    
    print(f'  Evaluating grid …', end=' ', flush=True)
    _te=time.time()
    C_sa  =SA(Z_cmp,T_cmp)
    C_slp =SLPNN(Z_cmp,T_cmp)
    print(f'done ({time.time()-_te:.1f}s)')

    E_abs=np.abs(C_slp-C_sa)
    L1_t=np.mean(E_abs,axis=0); L2_t=np.sqrt(np.mean(E_abs**2,axis=0))
    Li_t=np.max(E_abs,axis=0)
    L1_g=float(np.mean(L1_t)); L2_g=float(np.mean(L2_t)); RMSE=float(np.sqrt(np.mean(E_abs**2)))

    T_chk=np.linspace(T_BC_EVAL_MIN,T_END,400)
    r1=EPS_BC*SLPNN_dZ([1.],T_chk)[0]+W_S*SLPNN([1.],T_chk)[0]
    r2=EPS_BC*SLPNN_dZ([0.],T_chk)[0]+(W_S-B_REF)*SLPNN([0.],T_chk)[0]+B_REF*C_STAR
    ic_err=float(np.max(np.abs(SLPNN(Z_cmp,[0.])[:,0]-1.)))
    e_bc1=float(np.max(np.abs(r1))); e_bc0=float(np.max(np.abs(r2)))

    T_snaps=[0.1,0.5,1.0,2.0,T_END]
    snap_rows=[]
    for ts in T_snaps:
        idx=int(np.argmin(np.abs(T_cmp-ts)))
        snap_rows.append((T_cmp[idx],
            float(np.mean(E_abs[:,idx])),
            float(np.sqrt(np.mean(E_abs[:,idx]**2))),
            float(np.max(E_abs[:,idx]))))

    iz0=int(np.argmin(np.abs(Z_cmp-0.)))
    sa_pk=float(C_sa[iz0,:].max()); t_sa_pk=T_cmp[C_sa[iz0,:].argmax()]
    slp_pk=float(C_slp[iz0,:].max())

    print(f'\n  IC={ic_err:.2e}  BC1={e_bc1:.3e}  BC2={e_bc0:.3e}'
          f'  L²={L2_g:.4e}  RMSE={RMSE:.4e}  PeakΔ={abs(sa_pk-slp_pk):.2e}')

    RESULTS[ALPHA] = {
        'C_sa':C_sa,'C_slp':C_slp,'E_abs':E_abs,
        'L1_t':L1_t,'L2_t':L2_t,'Li_t':Li_t,
        'L1_g':L1_g,'L2_g':L2_g,'RMSE':RMSE,
        'ic_err':ic_err,'e_bc1':e_bc1,'e_bc0':e_bc0,
        'snap_rows':snap_rows,'all_hists':all_hists,
        'sa_pk':sa_pk,'slp_pk':slp_pk,'t_sa_pk':t_sa_pk,
        'E_qr':E_qr,'n_rank':n_rank,'s_sc':s_sc,'s_eff':s_eff,
        'kap':kap,
    }

# ═══════════════════════════════════════════════════════════════════════
#  6  COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  COMPARISON REPORT  —  α ∈ {0.50, 0.75, 0.90}')
print(SEP)
hdr=f"  {'Metric':<24} {'α=0.50':>14} {'α=0.75':>14} {'α=0.90':>14}"
print(hdr); print('  '+'-'*68)
metrics=[
    ('IC max error', 'ic_err',  '{:.2e}'),
    ('BC1 max res.',  'e_bc1',  '{:.3e}'),
    ('BC2 max res.',  'e_bc0',  '{:.3e}'),
    ('Global L²',     'L2_g',   '{:.4e}'),
    ('Global RMSE',   'RMSE',   '{:.4e}'),
    ('Peak Δ (Z=0)',  None,     '{:.2e}'),
    ('SVD rank',      'n_rank', '{}'),
    ('κ(J_sc)',        'kap',   '{:.1e}'),
    ('E_qr',          'E_qr',  '{:.4e}'),
]
for lbl,key,fmt in metrics:
    row=f'  {lbl:<24}'
    for a in ALPHA_LIST:
        if key=='n_rank': v=f"{RESULTS[a]['n_rank']}/225 (100%)"
        elif key is None: v=fmt.format(abs(RESULTS[a]['sa_pk']-RESULTS[a]['slp_pk']))
        else: v=fmt.format(RESULTS[a][key])
        row+=f' {v:>14}'
    print(row)
print(SEP)

# ═══════════════════════════════════════════════════════════════════════
#  7   FIGURES
# ═══════════════════════════════════════════════════════════════════════
LW=2.2; LW2=1.8
T_SNAPS_PLOT=[0.1,0.5,1.0,2.0,T_END]
C_snap=plt.cm.viridis(np.linspace(0.05,0.95,len(T_SNAPS_PLOT)))
C_zsel=plt.cm.plasma(np.linspace(0.05,0.95,5))
Z_sel=[0.05,0.25,0.50,0.75,0.95]

# ─── Figure 1 ─────────────────────────────
print('\n  [Fig 1/7]  Profile comparison α=0.50')
fig,ax=plt.subplots(figsize=(7,8))
R=RESULTS[0.50]
for ts,col in zip(T_SNAPS_PLOT,C_snap):
    idx=int(np.argmin(np.abs(T_cmp-ts)))
    ax.plot(R['C_sa'][:,idx], Z_cmp,color=col,ls='-', lw=LW,  zorder=3)
    ax.plot(R['C_slp'][:,idx],Z_cmp,color=col,ls='--',lw=LW2, alpha=.88,zorder=4)
ax.plot(C_INF(Z_cmp),Z_cmp,'k:',lw=1.8,alpha=.5)
prx=[Line2D([0],[0],c='k',ls='-', lw=LW, label='SA'),
     Line2D([0],[0],c='k',ls='--',lw=LW2,label='SLPNN')]
prx+=[Line2D([0],[0],c=col,ls='-',lw=LW,label=f't̂={ts:.1f}')
      for ts,col in zip(T_SNAPS_PLOT,C_snap)]
ax.legend(handles=prx,fontsize=8,loc='lower right')
ax.set_xlabel('Concentration  ĉ',fontsize=12)
ax.set_ylabel('Depth  Z  (0=bed, 1=surface)',fontsize=12)
ax.set_title(f'Figure 1 — Profile Comparison: SA (—) vs SLPNN (– –)\n'
    f'α=0.50  (strong sub-diffusion)   L²={R["L2_g"]:.3e}',fontsize=12)
ax.set_xlim(-0.02,1.50); ax.set_ylim(-0.01,1.01)
plt.tight_layout(); plt.show()

# ─── Figure 2─────────────────────────────
print('  [Fig 2/7]  Profile comparison α=0.75')
fig,ax=plt.subplots(figsize=(7,8))
R=RESULTS[0.75]
for ts,col in zip(T_SNAPS_PLOT,C_snap):
    idx=int(np.argmin(np.abs(T_cmp-ts)))
    ax.plot(R['C_sa'][:,idx], Z_cmp,color=col,ls='-', lw=LW,  zorder=3)
    ax.plot(R['C_slp'][:,idx],Z_cmp,color=col,ls='--',lw=LW2, alpha=.88,zorder=4)
ax.plot(C_INF(Z_cmp),Z_cmp,'k:',lw=1.8,alpha=.5)
prx=[Line2D([0],[0],c='k',ls='-', lw=LW, label='SA'),
     Line2D([0],[0],c='k',ls='--',lw=LW2,label='SLPNN')]
prx+=[Line2D([0],[0],c=col,ls='-',lw=LW,label=f't̂={ts:.1f}')
      for ts,col in zip(T_SNAPS_PLOT,C_snap)]
ax.legend(handles=prx,fontsize=8,loc='lower right')
ax.set_xlabel('Concentration  ĉ',fontsize=12)
ax.set_ylabel('Depth  Z  (0=bed, 1=surface)',fontsize=12)
ax.set_title(f'Figure 2 — Profile Comparison: SA (—) vs SLPNN (– –)\n'
    f'α=0.75  (moderate sub-diffusion)  L²={R["L2_g"]:.3e}',fontsize=12)
ax.set_xlim(-0.02,1.50); ax.set_ylim(-0.01,1.01)
plt.tight_layout(); plt.show()

# ─── Figure 3─────────────────────────────
print('  [Fig 3/7]  Profile comparison α=0.90')
fig,ax=plt.subplots(figsize=(7,8))
R=RESULTS[0.90]
for ts,col in zip(T_SNAPS_PLOT,C_snap):
    idx=int(np.argmin(np.abs(T_cmp-ts)))
    ax.plot(R['C_sa'][:,idx], Z_cmp,color=col,ls='-', lw=LW,  zorder=3)
    ax.plot(R['C_slp'][:,idx],Z_cmp,color=col,ls='--',lw=LW2, alpha=.88,zorder=4)
ax.plot(C_INF(Z_cmp),Z_cmp,'k:',lw=1.8,alpha=.5)
prx=[Line2D([0],[0],c='k',ls='-', lw=LW, label='SA'),
     Line2D([0],[0],c='k',ls='--',lw=LW2,label='SLPNN')]
prx+=[Line2D([0],[0],c=col,ls='-',lw=LW,label=f't̂={ts:.1f}')
      for ts,col in zip(T_SNAPS_PLOT,C_snap)]
ax.legend(handles=prx,fontsize=8,loc='lower right')
ax.set_xlabel('Concentration  ĉ',fontsize=12)
ax.set_ylabel('Depth  Z  (0=bed, 1=surface)',fontsize=12)
ax.set_title(f'Figure 3 — Profile Comparison: SA (—) vs SLPNN (– –)\n'
    f'α=0.90  (near-classical diffusion)  L²={R["L2_g"]:.3e}',fontsize=12)
ax.set_xlim(-0.02,1.50); ax.set_ylim(-0.01,1.01)
plt.tight_layout(); plt.show()

# ─── Figure 4 ───────────────────────
print('  [Fig 4/7]  α comparison at fixed times')
fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
for ax_i,ts in zip(axes,[0.5,1.0,2.0]):
    idx=int(np.argmin(np.abs(T_cmp-ts)))
    for a in ALPHA_LIST:
        R=RESULTS[a]; col=ALPHA_COLORS[a]
        ax_i.plot(R['C_sa'][:,idx], Z_cmp,color=col,ls='-', lw=LW+.3,
                  label=f'SA α={a}')
        ax_i.plot(R['C_slp'][:,idx],Z_cmp,color=col,ls='--',lw=LW2,alpha=.85,
                  label=f'SLPNN α={a}')
    ax_i.plot(C_INF(Z_cmp),Z_cmp,'k:',lw=1.6,alpha=.45)
    ax_i.set_xlabel('ĉ',fontsize=12)
    ax_i.set_title(f't̂ = {ts:.1f}',fontsize=12)
    ax_i.set_xlim(-0.02,1.40); ax_i.set_ylim(-0.01,1.01)
    ax_i.grid(True,alpha=.2)
    if ax_i==axes[0]: ax_i.set_ylabel('Z  (0=bed, 1=surface)',fontsize=12)
prx=[]
for a in ALPHA_LIST:
    prx.append(Line2D([0],[0],c=ALPHA_COLORS[a],ls='-',lw=LW+.3,label=f'SA  α={a}'))
    prx.append(Line2D([0],[0],c=ALPHA_COLORS[a],ls='--',lw=LW2,label=f'SLPNN  α={a}'))
prx.append(Line2D([0],[0],c='k',ls=':',lw=1.6,alpha=.5,label='Rouse ĉ∞'))
axes[1].legend(handles=prx,fontsize=8,loc='lower right',ncol=2)
fig.suptitle('Figure 4 — Effect of Fractional Order α on Concentration Profiles\n'
    'SA (—) vs SLPNN (– –)  |  α=0.50 (blue), α=0.75 (green), α=0.90 (red)',
    fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()

# ─── Figure 5─────────────────────
print('  [Fig 5/7]  Near-bed & convergence for all α')
fig,axes=plt.subplots(2,1,figsize=(10,10),sharex=True)
ax=axes[0]
for a in ALPHA_LIST:
    R=RESULTS[a]; col=ALPHA_COLORS[a]
    ax.plot(T_cmp,R['C_sa'][iz0,:],  color=col,ls='-', lw=LW+.3,label=f'SA  α={a}')
    ax.plot(T_cmp,R['C_slp'][iz0,:], color=col,ls='--',lw=LW2,alpha=.9)
    ax.axvline(R['t_sa_pk'],color=col,ls=':',lw=1.2,alpha=.6)
    ax.plot(R['t_sa_pk'],R['sa_pk'], marker='*',ms=14,color=col,zorder=10)
    ax.plot(R['t_sa_pk'],R['slp_pk'],marker='*',ms=10,color=col,
            markeredgecolor='w',markeredgewidth=1.5,zorder=11)
ax.axhline(1.,ls=':',c='gray',lw=1.1,alpha=.5,label='IC level  ĉ=1')
ax.axhline(float(C_INF(0)),ls='-.',c='k',lw=1.1,alpha=.4,
    label=f'ĉ∞(0)={float(C_INF(0)):.4f}')
ax.set_ylabel('ĉ at Z=0 (near-bed)',fontsize=12)
ax.set_title('Figure 5 — Near-Bed Fractional Memory Peak: All α Values\n'
    'SA (solid), SLPNN (dashed), Peak (★)',fontsize=12)
ax.legend(fontsize=9,ncol=2)


ax2=axes[1]
for a in ALPHA_LIST:
    R=RESULTS[a]; col=ALPHA_COLORS[a]
    ax2.semilogy(T_cmp,R['L2_t'],color=col,lw=LW,ls=ALPHA_LS[a],
                 label=f'α={a}  L²={R["L2_g"]:.3e}')
ax2.set_xlabel('t̂',fontsize=12)
ax2.set_ylabel('L²(Z) error vs SA',fontsize=12)
ax2.set_title('L² Error Norms vs Time for All α',fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True,which='both',alpha=.2)
ax2.set_xlim(0,T_END)
plt.tight_layout(); plt.show()

# ─── Figure 6─────────────────────
print('  [Fig 6/7]  Error contours for all α')
fig,axes=plt.subplots(1,3,figsize=(16,5.5))
for ax_i,a in zip(axes,ALPHA_LIST):
    R=RESULTS[a]; TM,ZM=np.meshgrid(T_cmp,Z_cmp)
    vmin=max(R['E_abs'].min(),5e-5); vmax=R['E_abs'].max()
    cf=ax_i.contourf(TM,ZM,R['E_abs'],
        levels=np.logspace(np.log10(vmin),np.log10(vmax),20),
        cmap='RdYlBu_r',norm=LogNorm(vmin=vmin,vmax=vmax),extend='both')
    cs=ax_i.contour(TM,ZM,R['E_abs'],levels=[1e-3,3e-3,1e-2],
        colors=['white'],linewidths=1.0,alpha=.7)
    ax_i.clabel(cs,fmt='%.0e',fontsize=8,inline=True)
    plt.colorbar(cf,ax=ax_i,label='|SLPNN−SA|',shrink=.92)
    ax_i.set_xlabel('t̂',fontsize=11); ax_i.set_ylabel('Z',fontsize=11)
    ax_i.set_title(f'α = {a}\nL²={R["L2_g"]:.3e}  RMSE={R["RMSE"]:.3e}',fontsize=11)
    for ts in [0.5,1.0,2.0]:
        ax_i.axvline(ts,ls=':',c='k',lw=.8,alpha=.4)
fig.suptitle('Figure 6 — Absolute Error |SLPNN − SA| (log scale)  for α = 0.50, 0.75, 0.90',
    fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()

# ─── Figure 7 ──────────────────────────────────
print('  [Fig 7/7]  Error norm comparison')
fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
norm_keys=[('L1_t','L1_g','L¹(Z) Error'),
           ('L2_t','L2_g','L²(Z) Error'),
           ('Li_t','Li_t','L^∞(Z) Error')]
for ax_i,(nk,gk,lbl) in zip(axes,norm_keys):
    for a in ALPHA_LIST:
        R=RESULTS[a]; col=ALPHA_COLORS[a]
        ax_i.semilogy(T_cmp,R[nk],color=col,lw=LW,ls=ALPHA_LS[a],
            label=f'α={a}  avg={np.mean(R[nk]):.3e}')
        ax_i.fill_between(T_cmp,R[nk],alpha=.06,color=col)
        for ts in [0.5,1.0,2.0]:
            idx=int(np.argmin(np.abs(T_cmp-ts)))
            ax_i.plot(T_cmp[idx],R[nk][idx],'o',color=col,ms=7,
                markeredgecolor='white',markeredgewidth=1.0,zorder=5)
    ax_i.set_ylabel(lbl,fontsize=11)
    ax_i.legend(fontsize=9); ax_i.grid(True,which='both',alpha=.2)
axes[-1].set_xlabel('t̂',fontsize=12)
axes[0].set_title('Figure 7 — Error Norms vs Time:  α=0.50 (blue), 0.75 (green), 0.90 (red)\n'
    'SA (—), SLPNN (– –)',fontsize=12)
plt.tight_layout(); plt.show()

# ─── Summary REPORT ──────────────────────────────────────────────
print(f'\n{SEP}')
print('  FINAL SUMMARY — SLPNN ALPHA COMPARISON')
print(SEP)
for a in ALPHA_LIST:
    R=RESULTS[a]
    print(f'  α = {a}:  IC={R["ic_err"]:.2e}  BC1={R["e_bc1"]:.3e}  '
          f'L²={R["L2_g"]:.4e}  RMSE={R["RMSE"]:.4e}  PeakΔ={abs(R["sa_pk"]-R["slp_pk"]):.2e}  '
          f'rank={R["n_rank"]}/225  κ={R["kap"]:.1e}')
print(SEP)
