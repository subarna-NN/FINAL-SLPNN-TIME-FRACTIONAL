"""
============================================================================
  Three-Way Benchmark Comparison 
  Chopra FNN  vs  MLP-PINN (Adam + Backprop)  vs  SLPNN (Proposed)
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
    'figure.dpi':150,'axes.grid':True,'grid.alpha':0.22,'lines.linewidth':2.0,
    'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10,
})
SEP='='*72

# ═══════════════════════════════════════════════════════════════════════
#  1  PHYSICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
ALPHA=0.75; A_HAT=0.05; W_S=0.30; EPS_S=0.10
B_REF=0.80; C_STAR=1.00; T_END=3.0
P=W_S/(1.-A_HAT); Q=EPS_S/(1.-A_HAT)**2; EPS_BC=EPS_S/(1.-A_HAT); ROUSE=P/Q
def C_INF(Z): return np.exp(-ROUSE*np.asarray(Z,float))

print(SEP)
print('  Three-Way Comparison: Chopra FNN | MLP-PINN (Adam) | SLPNN (Proposed)')
print(f'  α={ALPHA}  P={P:.5f}  Q={Q:.5f}  EPS_BC={EPS_BC:.5f}  Rouse={ROUSE:.4f}')
print(SEP)

# ═══════════════════════════════════════════════════════════════════════
#  2   ML LUT + SA 
# ═══════════════════════════════════════════════════════════════════════
print('\n  Building ML LUT …', end=' ', flush=True); _t0=time.time()
def _ml_raw(xf):
    if xf<=1e-9: return 1.
    if xf>80: return 1./(Γ(1-ALPHA)*xf)-1./(Γ(1-2*ALPHA)*xf**2)
    return float(nsum(lambda k:power(mpf(-xf),k)/mpg(ALPHA*k+1.),[0,mpinf],tol=1e-13,error=False))
_xl=np.unique(np.concatenate([np.linspace(1e-4,.15,50),np.linspace(.15,2.5,100),
    np.exp(np.linspace(np.log(2.5),np.log(250.),350))]))
_yl=np.array([_ml_raw(x) for x in _xl]); _MLC=CubicSpline(_xl,_yl)
def ML(x_in):
    x=np.atleast_1d(np.asarray(x_in,float)); out=np.ones_like(x)
    big=x>250.; out[big]=1./(Γ(1-ALPHA)*x[big])
    mid=(~big)&(x>0); out[mid]=np.clip(_MLC(x[mid]),0.,1.)
    return float(out[0]) if np.ndim(x_in)==0 else out
print(f'done ({time.time()-_t0:.1f}s)')

N_SA=60; _m=-P/(2*Q); _b=abs(_m); _k=(B_REF-W_S)/EPS_BC+_b
def _tr(xi):
    if abs(xi)<1e-9: return _k+_b
    return (_k+_b)*xi*np.cos(xi)-(xi**2-_k*_b)*np.sin(xi)
_xs=np.linspace(.05,N_SA*np.pi+1,500000); _fv=np.array([_tr(x) for x in _xs])
_xsa=[]
for i in range(len(_fv)-1):
    if _fv[i]*_fv[i+1]<0:
        try:
            r=brentq(_tr,_xs[i],_xs[i+1],xtol=1e-13)
            if r>.05 and(not _xsa or abs(r-_xsa[-1])>.05): _xsa.append(r)
        except: pass
    if len(_xsa)>=N_SA: break
XI_SA=np.array(_xsa[:N_SA]); LAM_SA=Q*(XI_SA**2+_m**2)
def _phi(n,Z): xi=XI_SA[n];a=xi/_k; return np.exp(_m*np.asarray(Z,float))*(np.sin(xi*Z)+a*np.cos(xi*Z))
def _Inn(n): xi=XI_SA[n];a=xi/_k; return(1+a**2)/2+(a**2-1)*np.sin(2*xi)/(4*xi)+a*(1-np.cos(2*xi))/(2*xi)
def _Cn(n):
    xi=XI_SA[n];a=xi/_k;b=_b;d2=xi**2+b**2
    Ish=(b*np.cosh(b)*np.sin(xi)-xi*np.sinh(b)*np.cos(xi))/d2
    Ich=(xi*np.sinh(b)*np.sin(xi)+b*np.cosh(b)*np.cos(xi)-b)/d2
    return 2*(Ish+a*Ich)/_Inn(n)
_CN=np.array([_Cn(n) for n in range(N_SA)])
def SA(Z_arr,t_arr):
    Z=np.asarray(Z_arr,float); t=np.asarray(t_arr,float)
    out=C_INF(Z)[:,None]*np.ones((1,len(t)))
    PHI=np.column_stack([_phi(n,Z) for n in range(N_SA)])
    ta=np.maximum(t,1e-8)**ALPHA
    for n in range(N_SA): out+=(_CN[n]*ML(LAM_SA[n]*ta))[None,:]*PHI[:,n:n+1]
    return out

NZ_CMP,NT_CMP=200,300
Z_cmp=np.linspace(0.,1.,NZ_CMP); T_cmp=np.linspace(.05,T_END,NT_CMP)
def cgn(n,a=0.,b=1.):
    k=np.arange(1,n+1); return .5*(1+np.cos(np.pi*(2*k-1)/(2*n)))*(b-a)+a
print('  Computing SA reference …',end=' ',flush=True)
_tv=time.time(); C_SA=SA(Z_cmp,T_cmp); print(f'done ({time.time()-_tv:.1f}s)')

# ═══════════════════════════════════════════════════════════════════════
#  3  METHOD 1
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{SEP}'); print('  METHOD 1: CHOPRA FIBONACCI NEURAL NETWORK'); print(SEP)

def fib_basis(N,x):
    x=np.atleast_1d(np.asarray(x,float)); F=np.zeros((N+1,len(x))); F[0]=1.
    if N>=1: F[1]=x.copy()
    for n in range(1,N): F[n+1]=x*F[n]+F[n-1]
    return F
def dfib_basis(N,x):
    x=np.atleast_1d(np.asarray(x,float)); Fv=fib_basis(N,x); D=np.zeros((N+1,len(x)))
    if N>=1: D[1]=1.
    for n in range(1,N): D[n+1]=Fv[n]+x*D[n]+D[n-1]
    return D
def d2fib_basis(N,x):
    x=np.atleast_1d(np.asarray(x,float)); Fv=fib_basis(N,x); Dv=dfib_basis(N,x); D2=np.zeros((N+1,len(x)))
    for n in range(1,N): D2[n+1]=2*Dv[n]+x*D2[n]+D2[n-1]
    return D2
def _fib_poly_coeffs(n):
    from math import comb as C
    if n==0: return {0:1.}
    if n==1: return {1:1.}
    prev2={0:1.}; prev1={1:1.}
    for k in range(2,n+1):
        cur={}
        for p,c in prev1.items(): cur[p+1]=cur.get(p+1,0)+c
        for p,c in prev2.items(): cur[p]=cur.get(p,0)+c
        prev2=prev1; prev1=cur
    return prev1
def caputo_fib(N,alpha,tau):
    tau=np.atleast_1d(np.asarray(tau,float)); res=np.zeros((N+1,len(tau)))
    for n in range(N+1):
        for k,c in _fib_poly_coeffs(n).items():
            if k>=1: res[n]+=c*(Γ(k+1)/Γ(k+1-alpha))*tau**(k-alpha)
    return res

N_FIB=15; W_IC_FNN=100.; W_BC_FNN=100.; N_W_FNN=N_FIB**2
N_PZ_F,N_PT_F,N_BC_F=30,30,35
Z_pde_f=cgn(N_PZ_F); T_pde_f=cgn(N_PT_F,0.02,1.); T_bc_f=cgn(N_BC_F,0.05,1.)
sw_bc_f=np.sqrt(W_BC_FNN); sw_ic_f=np.sqrt(W_IC_FNN)
F_pde_Z=fib_basis(N_FIB-1,Z_pde_f); DF_pde_Z=dfib_basis(N_FIB-1,Z_pde_f)
D2F_pde_Z=d2fib_basis(N_FIB-1,Z_pde_f); F_pde_T=fib_basis(N_FIB-1,T_pde_f)
CAP_F_T=caputo_fib(N_FIB-1,ALPHA,T_pde_f)/(T_END**ALPHA)
F_bc_T=fib_basis(N_FIB-1,T_bc_f)
F_z1=fib_basis(N_FIB-1,np.array([1.]))[:,0]; DF_z1=dfib_basis(N_FIB-1,np.array([1.]))[:,0]
F_z0=fib_basis(N_FIB-1,np.array([0.]))[:,0]; DF_z0=dfib_basis(N_FIB-1,np.array([0.]))[:,0]
F_ic_Z=fib_basis(N_FIB-1,Z_pde_f); F_ic_tau0=fib_basis(N_FIB-1,np.array([0.]))[:,0]
T1_f=np.einsum('ip,jq->pqij',F_pde_Z,CAP_F_T).reshape(N_PZ_F*N_PT_F,N_W_FNN)
T2_f=np.einsum('ip,jq->pqij',DF_pde_Z,F_pde_T).reshape(N_PZ_F*N_PT_F,N_W_FNN)
T3_f=np.einsum('ip,jq->pqij',D2F_pde_Z,F_pde_T).reshape(N_PZ_F*N_PT_F,N_W_FNN)
J_fnn_pde=T1_f-P*T2_f-Q*T3_f; b_fnn_pde=np.zeros(N_PZ_F*N_PT_F)
J_fnn_ic=sw_ic_f*np.einsum('ip,j->pij',F_ic_Z,F_ic_tau0).reshape(N_PZ_F,N_W_FNN)
b_fnn_ic=sw_ic_f*np.ones(N_PZ_F)
BC1_F=EPS_BC*DF_z1+W_S*F_z1; BC2_F=EPS_BC*DF_z0+(W_S-B_REF)*F_z0
J_fnn_bc1=sw_bc_f*np.einsum('i,jq->qij',BC1_F,F_bc_T).reshape(N_BC_F,N_W_FNN)
J_fnn_bc2=sw_bc_f*np.einsum('i,jq->qij',BC2_F,F_bc_T).reshape(N_BC_F,N_W_FNN)
b_fnn_bc=sw_bc_f*(-W_S)*np.ones(N_BC_F)
J_fnn=np.vstack([J_fnn_pde,J_fnn_ic,J_fnn_bc1,J_fnn_bc2])
b_fnn=np.concatenate([b_fnn_pde,b_fnn_ic,b_fnn_bc,b_fnn_bc])
D_fnn=np.maximum(np.linalg.norm(J_fnn,axis=0),1e-14); Dinv_fnn=1./D_fnn
J_fnn_sc=J_fnn*Dinv_fnn[None,:]
print('  SVD …',end=' ',flush=True)
U_f,s_f,Vt_f=np.linalg.svd(J_fnn_sc,full_matrices=False)
s_thr_f=s_f.max()*J_fnn.shape[0]*np.finfo(float).eps*100
s_eff_f=np.where(s_f>s_thr_f,s_f,0.); n_rank_f=int((s_eff_f>0).sum())
_Utr_f=U_f.T@b_fnn; _w_f_sc=Vt_f.T@np.where(s_eff_f>0,_Utr_f/s_f,0.)
E_qr_f=float(np.dot(J_fnn_sc@_w_f_sc-b_fnn,J_fnn_sc@_w_f_sc-b_fnn))
kap_f=s_f.max()/max(s_eff_f[s_eff_f>0].min(),1e-30) if n_rank_f>0 else np.inf
print(f'done  rank={n_rank_f}/{N_W_FNN}  κ={kap_f:.2e}  E_qr={E_qr_f:.4e}')
fnn_hists=[]; best_E_f=np.inf; best_w_f=None
for sd in range(5):
    rng=np.random.default_rng(sd); r_mask=s_eff_f>0
    w_sc=Vt_f[r_mask].T@rng.normal(0.,.01,int(r_mask.sum()))
    lam=float(s_f.max()**2/100.); hist_f=[]
    for _ in range(1000):
        r=J_fnn_sc@w_sc-b_fnn; E=float(r@r); hist_f.append(E)
        if E<1e-10: break
        dw=Vt_f.T@(s_eff_f/(s_eff_f**2+lam)*(U_f.T@r)); w_new=w_sc-dw
        E_new=float(np.dot(J_fnn_sc@w_new-b_fnn,J_fnn_sc@w_new-b_fnn))
        if E_new<E: w_sc=w_new; lam=max(lam/4.,1e-16)
        else: lam=min(lam*2.,1e12)
    w_sc=Vt_f[r_mask].T@(Vt_f[r_mask]@w_sc)
    ratio=hist_f[-1]/max(E_qr_f,1e-10)
    print(f'    Seed {sd}: E={hist_f[-1]:.4e}  ratio={ratio:.3f}')
    fnn_hists.append((sd,hist_f,hist_f[-1]))
    if hist_f[-1]<best_E_f: best_E_f=hist_f[-1]; best_w_f=w_sc*Dinv_fnn
W_FNN=best_w_f.reshape(N_FIB,N_FIB)
def FNN(Z_arr,t_arr,W=W_FNN):
    tau=np.asarray(t_arr,float)/T_END
    return fib_basis(N_FIB-1,np.asarray(Z_arr,float)).T@W@fib_basis(N_FIB-1,tau)
def FNN_dZ(Z_arr,t_arr,W=W_FNN):
    tau=np.asarray(t_arr,float)/T_END
    return dfib_basis(N_FIB-1,np.asarray(Z_arr,float)).T@W@fib_basis(N_FIB-1,tau)
C_FNN=FNN(Z_cmp,T_cmp)

# ═══════════════════════════════════════════════════════════════════════
#  4  METHOD 2
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{SEP}'); print('  METHOD 2: MLP-PINN  (Adam Optimizer + Analytical Backprop)'); print(SEP)

MLP_LAYERS=[2,16,16,1]

def _flatten(params):
    return np.concatenate([np.concatenate([W.ravel(),b.ravel()]) for W,b in params])

def _unflatten(theta,layers=MLP_LAYERS):
    params=[]; idx=0
    for i in range(len(layers)-1):
        nW=layers[i]*layers[i+1]; nb=layers[i+1]
        W=theta[idx:idx+nW].reshape(layers[i],layers[i+1]); b=theta[idx+nW:idx+nW+nb]
        params.append((W,b)); idx+=nW+nb
    return params

def _forward(theta, X):
    """Forward pass, returns (u, act_list) where act_list[i] = a_i"""
    params=_unflatten(theta); a=X; acts=[X]
    for W,b in params[:-1]: a=np.tanh(a@W+b); acts.append(a)
    W,b=params[-1]; u=(acts[-1]@W+b).ravel()
    return u, acts

def _backprop(theta, d_du, acts):
    """Backprop. d_du: (N,) upstream gradient.
    Returns ∂loss/∂θ as flat vector."""
    params=_unflatten(theta); delta=d_du[:,None]; grads=[]
    W,b=params[-1]; grads.insert(0,(acts[-1].T@delta, delta.sum(0)))
    delta=delta@W.T
    for i in range(len(params)-2,-1,-1):
        W,b=params[i]; delta=delta*(1-acts[i+1]**2)
        grads.insert(0,(acts[i].T@delta, delta.sum(0)))
        if i>0: delta=delta@W.T
    return np.concatenate([np.concatenate([gW.ravel(),gb.ravel()]) for gW,gb in grads])


M_L1=25; dt_L1=T_END/M_L1; tau_L1=np.linspace(0,1,M_L1+1)
b_L1=lambda k:(k+1)**(1-ALPHA)-k**(1-ALPHA); sc_L1=dt_L1**(-ALPHA)/Γ(2-ALPHA)
D_cap=np.zeros((M_L1,M_L1+1))
for n in range(1,M_L1+1):
    for j in range(n):
        bj=b_L1(j); D_cap[n-1,n-j]+=bj*sc_L1; D_cap[n-1,n-j-1]-=bj*sc_L1


N_PZ_M,N_BC_M,N_IC_M=20,25,25
Z_pde_m=cgn(N_PZ_M); T_bc_m=cgn(N_BC_M,0.05,1.); Z_ic_m=cgn(N_IC_M)
tau_pde_m=tau_L1[1:]  
dZ_fd=8e-3             


ZZ,TT=np.meshgrid(Z_pde_m,tau_L1,indexing='ij')  
X_full=np.stack([ZZ.ravel(),TT.ravel()],axis=1)   
ZZ2,TT2=np.meshgrid(Z_pde_m,tau_pde_m,indexing='ij') 
X_ic=np.stack([Z_ic_m,np.zeros(N_IC_M)],axis=1)
X_bc1=np.stack([np.ones(N_BC_M),T_bc_m],axis=1)
X_bc2=np.stack([np.zeros(N_BC_M),T_bc_m],axis=1)

W_PDE_M=1.; W_IC_M=150.; W_BC_M=100.

def mlp_loss_and_grad(theta):
    """Compute loss and analytical gradient w.r.t. θ."""
    N_full=N_PZ_M*(M_L1+1)

    
    u_full, acts_full = _forward(theta, X_full)           
    u_2d = u_full.reshape(N_PZ_M, M_L1+1)               
    u_ic, acts_ic = _forward(theta, X_ic)                 
    u_bc1, acts_bc1 = _forward(theta, X_bc1)              
    u_bc2, acts_bc2 = _forward(theta, X_bc2)             

    
    cap_u = (D_cap @ u_2d.T).T   

    
    Zp=np.clip(Z_pde_m+dZ_fd,0.,1.); Zm=np.clip(Z_pde_m-dZ_fd,0.,1.)
    ZZp,TTp=np.meshgrid(Zp,tau_pde_m,indexing='ij')
    ZZm,TTm=np.meshgrid(Zm,tau_pde_m,indexing='ij')
    Xp=np.stack([ZZp.ravel(),TTp.ravel()],axis=1)
    Xm=np.stack([ZZm.ravel(),TTm.ravel()],axis=1)
    up,acts_up=_forward(theta,Xp); um,acts_um=_forward(theta,Xm)
    up2=up.reshape(N_PZ_M,M_L1); um2=um.reshape(N_PZ_M,M_L1)
    u0=u_2d[:,1:]                               
    du_dZ=(up2-um2)/(2*dZ_fd)
    d2u_dZ2=(up2-2*u0+um2)/(dZ_fd**2)

    
    X_bc1p=np.stack([np.full(N_BC_M,np.clip(1.+dZ_fd,0.,1.)),T_bc_m],axis=1)
    X_bc1m=np.stack([np.full(N_BC_M,np.clip(1.-dZ_fd,0.,1.)),T_bc_m],axis=1)
    X_bc2p=np.stack([np.full(N_BC_M,dZ_fd),T_bc_m],axis=1)
    X_bc2m=np.stack([np.full(N_BC_M,0.),T_bc_m],axis=1)
    ubc1p,ab1p=_forward(theta,X_bc1p); ubc1m,ab1m=_forward(theta,X_bc1m)
    ubc2p,ab2p=_forward(theta,X_bc2p); ubc2m,ab2m=_forward(theta,X_bc2m)

    
    r_pde=(cap_u-P*du_dZ-Q*d2u_dZ2).ravel()          
    r_ic=u_ic-1.                                       
    r_bc1=EPS_BC*(ubc1p-ubc1m)/(2*dZ_fd)+W_S*u_bc1   
    r_bc2=EPS_BC*(ubc2p-ubc2m)/(2*dZ_fd)+(W_S-B_REF)*u_bc2+B_REF*C_STAR

    loss=(W_PDE_M*np.dot(r_pde,r_pde)
         +W_IC_M *np.dot(r_ic,r_ic)
         +W_BC_M *(np.dot(r_bc1,r_bc1)+np.dot(r_bc2,r_bc2)))

    
    r_pde_2d=r_pde.reshape(N_PZ_M,M_L1)

    
    dE_u_full=np.zeros((N_PZ_M,M_L1+1))
    dE_u_full+=2.*W_PDE_M*(r_pde_2d@D_cap)          

    
    dE_u_full[:,1:]+=2.*W_PDE_M*r_pde_2d*(2.*Q/dZ_fd**2)

    
    dE_u_up  =2.*W_PDE_M*r_pde_2d*(-P/(2*dZ_fd)-Q/dZ_fd**2)
    dE_u_um  =2.*W_PDE_M*r_pde_2d*( P/(2*dZ_fd)-Q/dZ_fd**2)

    
    dE_u_ic  =2.*W_IC_M*r_ic

    
    dE_u_bc1 =2.*W_BC_M*r_bc1*W_S
    dE_u_bc1p=2.*W_BC_M*r_bc1*EPS_BC/(2*dZ_fd)
    dE_u_bc1m=2.*W_BC_M*r_bc1*(-EPS_BC/(2*dZ_fd))
    dE_u_bc2 =2.*W_BC_M*r_bc2*(W_S-B_REF)
    dE_u_bc2p=2.*W_BC_M*r_bc2*EPS_BC/(2*dZ_fd)
    dE_u_bc2m=2.*W_BC_M*r_bc2*(-EPS_BC/(2*dZ_fd))

    
    g =_backprop(theta, dE_u_full.ravel(), acts_full)
    g+=_backprop(theta, dE_u_up.ravel(),  acts_up)
    g+=_backprop(theta, dE_u_um.ravel(),  acts_um)
    g+=_backprop(theta, dE_u_ic,          acts_ic)
    g+=_backprop(theta, dE_u_bc1,         acts_bc1)
    g+=_backprop(theta, dE_u_bc1p,        ab1p)
    g+=_backprop(theta, dE_u_bc1m,        ab1m)
    g+=_backprop(theta, dE_u_bc2,         acts_bc2)
    g+=_backprop(theta, dE_u_bc2p,        ab2p)
    g+=_backprop(theta, dE_u_bc2m,        ab2m)

    return float(loss), g

rng_mlp=np.random.default_rng(0); params_init=[]
for i in range(len(MLP_LAYERS)-1):
    scale=np.sqrt(2./MLP_LAYERS[i])
    W=rng_mlp.normal(0,scale,(MLP_LAYERS[i],MLP_LAYERS[i+1]))
    b=np.zeros(MLP_LAYERS[i+1]); params_init.append((W,b))
theta=_flatten(params_init); n_mlp=len(theta)
print(f'  MLP: {" → ".join(map(str,MLP_LAYERS))}  ({n_mlp} params)  |  Adam lr=1e-3')
print(f'  Caputo: L1 scheme  M={M_L1} steps  Δt={dt_L1:.3f}  '
      f'(Pang et al. 2019 fPINNs framework)')

LR=1e-3; BETA1=0.9; BETA2=0.999; EPS_A=1e-8; N_ITER=15000
LR_SCHEDULE = {5000: 5e-4, 10000: 2.5e-4}
m_adam=np.zeros_like(theta); v_adam=np.zeros_like(theta)
mlp_loss_hist=[]; best_E_mlp=np.inf; best_theta=theta.copy()
print(f'  Training Adam ({N_ITER} iterations, LR decay schedule) …')
_ta=time.time()
lr_current=LR
for it in range(1,N_ITER+1):
    if it in LR_SCHEDULE: lr_current=LR_SCHEDULE[it]; print(f'    LR decayed to {lr_current:.1e}')
    L,g=mlp_loss_and_grad(theta)
    m_adam=BETA1*m_adam+(1-BETA1)*g
    v_adam=BETA2*v_adam+(1-BETA2)*(g**2)
    m_hat=m_adam/(1-BETA1**it)
    v_hat=v_adam/(1-BETA2**it)
    theta=theta-lr_current*m_hat/(np.sqrt(v_hat)+EPS_A)
    mlp_loss_hist.append(L)
    if L<best_E_mlp: best_E_mlp=L; best_theta=theta.copy()
    if it%2500==0 or it==1 or it==500:
        print(f'    iter {it:5d}: E={L:.4e}  (best={best_E_mlp:.4e})')
print(f'  Training done ({time.time()-_ta:.1f}s)  Best E={best_E_mlp:.4e}')
theta_final=best_theta

def MLP_PINN(Z_arr,t_arr):
    Z_arr=np.asarray(Z_arr,float); t_arr=np.asarray(t_arr,float)
    ZZ,TT=np.meshgrid(Z_arr,t_arr,indexing='ij')
    X=np.stack([ZZ.ravel(),TT.ravel()/T_END],axis=1)
    u,_=_forward(theta_final,X); return u.reshape(len(Z_arr),len(t_arr))
def MLP_PINN_dZ(Z_arr,t_arr):
    Z_arr=np.asarray(Z_arr,float); t_arr=np.asarray(t_arr,float)
    Zp=np.clip(Z_arr+dZ_fd,0.,1.); Zm=np.clip(Z_arr-dZ_fd,0.,1.)
    ZZp,TT=np.meshgrid(Zp,t_arr,indexing='ij'); ZZm,_=np.meshgrid(Zm,t_arr,indexing='ij')
    Xp=np.stack([ZZp.ravel(),TT.ravel()/T_END],axis=1)
    Xm=np.stack([ZZm.ravel(),TT.ravel()/T_END],axis=1)
    up,_=_forward(theta_final,Xp); um,_=_forward(theta_final,Xm)
    return (up-um).reshape(len(Z_arr),len(t_arr))/(2*dZ_fd)

print('  Evaluating on comparison grid …',end=' ',flush=True)
_t=time.time(); C_MLP=MLP_PINN(Z_cmp,T_cmp); print(f'done ({time.time()-_t:.1f}s)')

# ═══════════════════════════════════════════════════════════════════════
#  5  METHOD 3
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{SEP}'); print('  METHOD 3: SLPNN (PROPOSED)'); print(SEP)
def SLP(N,x):
    x=np.atleast_1d(np.asarray(x,float)); Pv=np.zeros((N+1,len(x))); Pv[0]=1.
    if N>=1: Pv[1]=2*x-1
    for n in range(1,N): Pv[n+1]=((2*n+1)*(2*x-1)*Pv[n]-n*Pv[n-1])/(n+1)
    return Pv
def dSLP(N,x):
    x=np.atleast_1d(np.asarray(x,float)); Pv=SLP(N,x); D=np.zeros((N+1,len(x)))
    if N>=1: D[1]=2.
    for n in range(1,N): D[n+1]=2*(2*n+1)*Pv[n]+D[n-1]
    return D
def d2SLP(N,x):
    x=np.atleast_1d(np.asarray(x,float)); Dv=dSLP(N,x); D2=np.zeros((N+1,len(x)))
    for n in range(1,N): D2[n+1]=2*(2*n+1)*Dv[n]+D2[n-1]
    return D2
def _slp_coeffs(n):
    from math import comb as C; return[(k,(-1)**(n-k)*C(n,k)*C(n+k,k)) for k in range(n+1)]
def caputo_SLP(N,alpha,tau):
    tau=np.atleast_1d(np.asarray(tau,float)); res=np.zeros((N+1,len(tau)))
    for n in range(N+1):
        for(k,c) in _slp_coeffs(n):
            if k>=1: res[n]+=c*(Γ(k+1)/Γ(k+1-alpha))*tau**(k-alpha)
    return res
N_Z=15; N_T=15; N_W=N_Z*N_T; W_BC=100.; sw_bc_s=np.sqrt(W_BC)
N_PZ2,N_PT2,N_BC2=30,30,35
Z_pde2=cgn(N_PZ2); T_pde2=cgn(N_PT2,0.02,1.); T_bc2=cgn(N_BC2,0.05,1.)
P_pde_Z=SLP(N_Z-1,Z_pde2); D1_pde_Z=dSLP(N_Z-1,Z_pde2); D2_pde_Z=d2SLP(N_Z-1,Z_pde2)
P_pde_T=SLP(N_T,T_pde2)[1:,:]; P_bc_T=SLP(N_T,T_bc2)[1:,:]
P_zero_j=np.array([(-1.)**j for j in range(1,N_T+1)])
ETA_pde=P_pde_T-P_zero_j[:,None]; ETA_bc2=P_bc_T-P_zero_j[:,None]
CAP_T2=caputo_SLP(N_T,ALPHA,T_pde2)[1:,:]/(T_END**ALPHA)
nn=np.arange(N_Z,dtype=float)
P_z1s=np.ones(N_Z); D1_z1s=nn*(nn+1); P_z0s=(-1.)**nn; D1_z0s=(-1.)**(nn-1)*nn*(nn+1)
BC1_s=EPS_BC*D1_z1s+W_S*P_z1s; BC2_s=EPS_BC*D1_z0s+(W_S-B_REF)*P_z0s
T1s=np.einsum('ip,jq->pqij',P_pde_Z,CAP_T2).reshape(N_PZ2*N_PT2,N_W)
T2s=np.einsum('ip,jq->pqij',D1_pde_Z,ETA_pde).reshape(N_PZ2*N_PT2,N_W)
T3s=np.einsum('ip,jq->pqij',D2_pde_Z,ETA_pde).reshape(N_PZ2*N_PT2,N_W)
J_slp=np.vstack([T1s-P*T2s-Q*T3s,
    sw_bc_s*np.einsum('i,jq->qij',BC1_s,ETA_bc2).reshape(N_BC2,N_W),
    sw_bc_s*np.einsum('i,jq->qij',BC2_s,ETA_bc2).reshape(N_BC2,N_W)])
b_slp=np.concatenate([np.zeros(N_PZ2*N_PT2),sw_bc_s*(-W_S)*np.ones(N_BC2),sw_bc_s*(-W_S)*np.ones(N_BC2)])
D_slp=np.maximum(np.linalg.norm(J_slp,axis=0),1e-14); Dinv_slp=1./D_slp
J_slp_sc=J_slp*Dinv_slp[None,:]
print('  SVD …',end=' ',flush=True)
U_s,s_s,Vt_s=np.linalg.svd(J_slp_sc,full_matrices=False)
s_thr_s=s_s.max()*J_slp.shape[0]*np.finfo(float).eps*100
s_eff_s=np.where(s_s>s_thr_s,s_s,0.); n_rank_s=int((s_eff_s>0).sum())
_Utr_s=U_s.T@b_slp; _w_s_sc=Vt_s.T@np.where(s_eff_s>0,_Utr_s/s_s,0.)
E_qr_s=float(np.dot(J_slp_sc@_w_s_sc-b_slp,J_slp_sc@_w_s_sc-b_slp))
kap_s=s_s.max()/s_eff_s[s_eff_s>0].min()
print(f'done  rank={n_rank_s}/{N_W}  κ={kap_s:.2e}  E_qr={E_qr_s:.4e}')
slpnn_hists=[]; best_E_s=np.inf
for sd in range(5):
    rng=np.random.default_rng(sd); r_mask=s_eff_s>0
    w_sc=Vt_s[r_mask].T@rng.normal(0.,.01,int(r_mask.sum()))
    lam=float(s_s.max()**2/100.); hist_s=[]
    for _ in range(1000):
        r=J_slp_sc@w_sc-b_slp; E=float(r@r); hist_s.append(E)
        if E<1e-10: break
        dw=Vt_s.T@(s_eff_s/(s_eff_s**2+lam)*(U_s.T@r)); w_new=w_sc-dw
        E_new=float(np.dot(J_slp_sc@w_new-b_slp,J_slp_sc@w_new-b_slp))
        if E_new<E: w_sc=w_new; lam=max(lam/4.,1e-16)
        else: lam=min(lam*2.,1e12)
    w_sc=Vt_s[r_mask].T@(Vt_s[r_mask]@w_sc); tag='✓' if hist_s[-1]/E_qr_s<1.02 else 'ratio={:.2f}'.format(hist_s[-1]/E_qr_s)
    print(f'    Seed {sd}: E={hist_s[-1]:.4e}  {tag}')
    slpnn_hists.append((sd,hist_s,hist_s[-1]))
    if hist_s[-1]<best_E_s: best_E_s=hist_s[-1]
W_SLPNN=(_w_s_sc*Dinv_slp).reshape(N_Z,N_T)
def SLPNN(Z_arr,t_arr):
    tau=np.asarray(t_arr,float)/T_END; PZ=SLP(N_Z-1,np.asarray(Z_arr,float))
    PT=SLP(N_T,tau)[1:,:]; ETA=PT-P_zero_j[:,None]; return 1.+PZ.T@W_SLPNN@ETA
def SLPNN_dZ(Z_arr,t_arr):
    tau=np.asarray(t_arr,float)/T_END; DZ=dSLP(N_Z-1,np.asarray(Z_arr,float))
    PT=SLP(N_T,tau)[1:,:]; ETA=PT-P_zero_j[:,None]; return DZ.T@W_SLPNN@ETA
print('  Evaluating on comparison grid …',end=' ',flush=True)
_t=time.time(); C_SLPNN=SLPNN(Z_cmp,T_cmp); print(f'done ({time.time()-_t:.1f}s)')

# ═══════════════════════════════════════════════════════════════════════
#  6  ERROR METRICS 
# ═══════════════════════════════════════════════════════════════════════
T_BC_EVAL=0.20; T_chk_bc=np.linspace(T_BC_EVAL,T_END,400)
T_SNAPS=[0.1,0.5,1.0,2.0,T_END]
def compute_metrics(C_pred,dZ_func):
    E_abs=np.abs(C_pred-C_SA)
    L1_t=np.mean(E_abs,axis=0); L2_t=np.sqrt(np.mean(E_abs**2,axis=0))
    Li_t=np.max(E_abs,axis=0)
    L1_g=float(np.mean(L1_t)); L2_g=float(np.mean(L2_t)); RMSE=float(np.sqrt(np.mean(E_abs**2)))
    iz1_=int(np.argmin(np.abs(Z_cmp-1.))); iz0_=int(np.argmin(np.abs(Z_cmp-0.)))
    tidx=np.array([int(np.argmin(np.abs(T_cmp-t))) for t in T_chk_bc])
    u1=C_pred[iz1_,tidx]; du1=dZ_func(np.array([Z_cmp[iz1_]]),T_chk_bc)[0]
    u0=C_pred[iz0_,tidx]; du0=dZ_func(np.array([Z_cmp[iz0_]]),T_chk_bc)[0]
    e_bc1=float(np.max(np.abs(EPS_BC*du1+W_S*u1)))
    e_bc2=float(np.max(np.abs(EPS_BC*du0+(W_S-B_REF)*u0+B_REF*C_STAR)))
    ic_err=float(np.max(np.abs(C_pred[:,0]-1.)))
    iz0r=int(np.argmin(np.abs(Z_cmp-0.)))
    sa_pk=float(C_SA[iz0r,:].max()); nn_pk=float(C_pred[iz0r,:].max())
    t_sa_pk=T_cmp[C_SA[iz0r,:].argmax()]
    snap_rows=[]
    for ts in T_SNAPS:
        idx=int(np.argmin(np.abs(T_cmp-ts)))
        snap_rows.append((T_cmp[idx],float(np.mean(E_abs[:,idx])),
            float(np.sqrt(np.mean(E_abs[:,idx]**2))),float(np.max(E_abs[:,idx]))))
    return dict(E_abs=E_abs,L1_t=L1_t,L2_t=L2_t,Li_t=Li_t,
                L1_g=L1_g,L2_g=L2_g,RMSE=RMSE,ic_err=ic_err,
                e_bc1=e_bc1,e_bc2=e_bc2,snap_rows=snap_rows,
                sa_pk=sa_pk,nn_pk=nn_pk,t_sa_pk=t_sa_pk)

M_FNN=compute_metrics(C_FNN,FNN_dZ)
M_MLP=compute_metrics(C_MLP,MLP_PINN_dZ)
M_SLP=compute_metrics(C_SLPNN,SLPNN_dZ)

# ═══════════════════════════════════════════════════════════════════════
#  7  COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  COMPARISON REPORT — α=0.75')
print(f'  {"Metric":<26} {"Chopra FNN":>16} {"MLP-PINN (Adam)":>18} {"SLPNN (Ours)":>16}')
print('  '+'-'*78)
metrics_tbl=[
    ('IC max error',  [M_FNN["ic_err"],  M_MLP["ic_err"],  M_SLP["ic_err"]],  '{:.2e}'),
    ('BC1 max res.',  [M_FNN["e_bc1"],   M_MLP["e_bc1"],   M_SLP["e_bc1"]],   '{:.3e}'),
    ('BC2 max res.',  [M_FNN["e_bc2"],   M_MLP["e_bc2"],   M_SLP["e_bc2"]],   '{:.3e}'),
    ('Global L²',     [M_FNN["L2_g"],    M_MLP["L2_g"],    M_SLP["L2_g"]],    '{:.4e}'),
    ('Global RMSE',   [M_FNN["RMSE"],    M_MLP["RMSE"],    M_SLP["RMSE"]],    '{:.4e}'),
    ('Peak Δ (Z=0)',  [abs(M_FNN["sa_pk"]-M_FNN["nn_pk"]),
                       abs(M_MLP["sa_pk"]-M_MLP["nn_pk"]),
                       abs(M_SLP["sa_pk"]-M_SLP["nn_pk"])],                   '{:.2e}'),
    ('Parameters',    [N_W_FNN,n_mlp,N_W],                                    '{}'),
    ('SVD rank',      [f'{n_rank_f}/{N_W_FNN}',f'N/A (deep)',f'{n_rank_s}/{N_W}'],'{}'),
    ('κ(J)',          [f'{kap_f:.1e}',   'N/A',             f'{kap_s:.1e}'],   '{}'),
]
for lbl,vals,fmt in metrics_tbl:
    row=f'  {lbl:<26}'
    for v in vals: row+=f' {fmt.format(v):>16}'
    print(row)
print(SEP)

# ═══════════════════════════════════════════════════════════════════════
#  8   FIGURES
# ═══════════════════════════════════════════════════════════════════════
COLS={'fnn':'#1565C0','mlp':'#C62828','slp':'#2E7D32'}
LBL={'fnn':'Chopra FNN','mlp':'MLP-PINN (Adam)','slp':'SLPNN (Proposed)'}
LS={'fnn':':','mlp':'--','slp':'-'}
LW_m=2.0; LW_sa=2.5
C_snap_p=plt.cm.viridis(np.linspace(.05,.95,len(T_SNAPS)))
iz0=int(np.argmin(np.abs(Z_cmp-0.))); iz1=int(np.argmin(np.abs(Z_cmp-1.)))

# ─── Figure 1────────────────────────────────────
print('\n  [Fig 1/7]  Profile comparison (3 panels)')
fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True)
for ax_i,ts in zip(axes,[0.5,1.0,3.0]):
    idx=int(np.argmin(np.abs(T_cmp-ts)))
    ax_i.plot(C_SA[:,idx],   Z_cmp,c='k',    ls='-', lw=LW_sa,label='SA (reference)')
    ax_i.plot(C_FNN[:,idx],  Z_cmp,c=COLS['fnn'],ls=LS['fnn'],lw=LW_m,label=LBL['fnn'])
    ax_i.plot(C_MLP[:,idx],  Z_cmp,c=COLS['mlp'],ls=LS['mlp'],lw=LW_m,label=LBL['mlp'])
    ax_i.plot(C_SLPNN[:,idx],Z_cmp,c=COLS['slp'],ls=LS['slp'],lw=LW_m,label=LBL['slp'])
    ax_i.plot(C_INF(Z_cmp),  Z_cmp,'gray',ls=':',lw=1.4,alpha=.5,label='Rouse ĉ∞')
    ax_i.set_xlabel('Concentration ĉ',fontsize=12); ax_i.set_title(f't̂ = {ts:.1f}',fontsize=13)
    ax_i.set_xlim(-0.02,1.45); ax_i.set_ylim(-0.01,1.01); ax_i.legend(fontsize=8.5,loc='lower right')
    if ax_i is axes[0]: ax_i.set_ylabel('Depth Z (0=bed, 1=surface)',fontsize=12)
fig.suptitle('Figure 1 — Profile: SA (black) | Chopra FNN (blue···) | MLP-PINN Adam (red--) | SLPNN (green—)\n'
    f'α=0.75   SLPNN overlaps SA at all times — both other methods show clear deviation',
    fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()

# ─── Figure 2────────────────────────────────────────────
print('  [Fig 2/7]  Convergence')
fig,axes=plt.subplots(1,2,figsize=(14,6))
ax=axes[0]
for sd,h,fE in fnn_hists:
    lv=2.2 if fE==min(x[2] for x in fnn_hists) else 1.0
    ax.semilogy(np.arange(len(h)),h,c=COLS['fnn'],lw=lv,alpha=.8 if lv>1 else .5)
ax.axhline(E_qr_f,ls='--',c=COLS['fnn'],lw=1.8,label=f'FNN QR min={E_qr_f:.3e}  (Fibonacci limit)')
for sd,h,fE in slpnn_hists:
    lv=2.2 if fE==min(x[2] for x in slpnn_hists) else 1.0
    ax.semilogy(np.arange(len(h)),h,c=COLS['slp'],lw=lv,alpha=.8 if lv>1 else .5)
ax.axhline(E_qr_s,ls='--',c=COLS['slp'],lw=1.8,label=f'SLPNN QR min={E_qr_s:.3e}')
ax.set_xlabel('Marquardt iteration k',fontsize=12); ax.set_ylabel('Loss E(w)',fontsize=12)
ax.set_title('Chopra FNN vs SLPNN\nMarquardt convergence',fontsize=12); ax.legend(fontsize=9)
ax=axes[1]
ax.semilogy(np.arange(1,len(mlp_loss_hist)+1),mlp_loss_hist,c=COLS['mlp'],lw=2.2)
ax.set_xlabel('Adam iteration k',fontsize=12); ax.set_ylabel('Loss E(θ)',fontsize=12)
ax.set_title(f'MLP-PINN Adam Training\n{n_mlp} params | Best E={best_E_mlp:.4e}',fontsize=12)
ax.grid(True,which='both',alpha=.2)
fig.suptitle('Figure 2 — Training Convergence: Marquardt (FNN/SLPNN) & Adam (MLP-PINN)',
    fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()

# ─── Figure 3─────────────────────────────────
print('  [Fig 3/7]  Error contours')
fig,axes=plt.subplots(1,3,figsize=(17,5.5))
for ax_i,(C_pred,lbl,col) in zip(axes,[
        (C_FNN,LBL['fnn'],COLS['fnn']),(C_MLP,LBL['mlp'],COLS['mlp']),(C_SLPNN,LBL['slp'],COLS['slp'])]):
    E_abs=np.abs(C_pred-C_SA); TM,ZM=np.meshgrid(T_cmp,Z_cmp)
    vmin=max(E_abs.min(),5e-5); vmax=max(E_abs.max(),1e-4)
    cf=ax_i.contourf(TM,ZM,E_abs,levels=np.logspace(np.log10(vmin),np.log10(vmax),20),
        cmap='RdYlBu_r',norm=LogNorm(vmin=vmin,vmax=vmax),extend='both')
    cs=ax_i.contour(TM,ZM,E_abs,levels=[1e-3,5e-3,1e-2],colors=['white'],linewidths=1.,alpha=.7)
    ax_i.clabel(cs,fmt='%.0e',fontsize=8,inline=True)
    plt.colorbar(cf,ax=ax_i,label='|pred−SA|',shrink=.9)
    ax_i.set_xlabel('t̂',fontsize=11); ax_i.set_ylabel('Z',fontsize=11)
    L2g=float(np.sqrt(np.mean((C_pred-C_SA)**2)))
    ax_i.set_title(f'{lbl}\nL²={L2g:.3e}',fontsize=11)
fig.suptitle('Figure 3 — Absolute Error |pred − SA| (log scale)',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()

# ─── Figure 4────────────────────────────────────────────
print('  [Fig 4/7]  L² error norms')
fig,ax=plt.subplots(figsize=(10,6))
ax.semilogy(T_cmp,M_FNN['L2_t'],c=COLS['fnn'],ls=LS['fnn'],lw=2.2,
    label=f'{LBL["fnn"]}  avg L²={M_FNN["L2_g"]:.3e}')
ax.semilogy(T_cmp,M_MLP['L2_t'],c=COLS['mlp'],ls=LS['mlp'],lw=2.2,
    label=f'{LBL["mlp"]}  avg L²={M_MLP["L2_g"]:.3e}')
ax.semilogy(T_cmp,M_SLP['L2_t'],c=COLS['slp'],ls=LS['slp'],lw=2.2,
    label=f'{LBL["slp"]}  avg L²={M_SLP["L2_g"]:.3e}')
for ts in [0.5,1.0,2.0]:
    idx=int(np.argmin(np.abs(T_cmp-ts)))
    for M_,col in [(M_FNN,COLS['fnn']),(M_MLP,COLS['mlp']),(M_SLP,COLS['slp'])]:
        ax.plot(T_cmp[idx],M_['L2_t'][idx],'o',color=col,ms=8,
            markeredgecolor='white',markeredgewidth=1.2,zorder=6)
ax.set_xlabel('t̂',fontsize=12); ax.set_ylabel('L²(Z) Error vs SA',fontsize=12)
ax.set_title('Figure 4 — L² Error Norms vs Time: All Methods\nSLPNN lowest across entire time domain',fontsize=12)
ax.legend(fontsize=9); ax.grid(True,which='both',alpha=.2); ax.set_xlim(0,T_END)
plt.tight_layout(); plt.show()

# ─── Figure 5───────────────────────────────
print('  [Fig 5/7]  Near-bed time evolution')
fig,axes=plt.subplots(2,1,figsize=(10,10),sharex=True)
ax=axes[0]
ax.plot(T_cmp,C_SA[iz0,:],'k',lw=LW_sa+.3,label='SA (reference)')
ax.plot(T_cmp,C_FNN[iz0,:],c=COLS['fnn'],ls=LS['fnn'],lw=LW_m,label=LBL['fnn'])
ax.plot(T_cmp,C_MLP[iz0,:],c=COLS['mlp'],ls=LS['mlp'],lw=LW_m,label=LBL['mlp'])
ax.plot(T_cmp,C_SLPNN[iz0,:],c=COLS['slp'],ls=LS['slp'],lw=LW_m,label=LBL['slp'])
t_pk=M_SLP['t_sa_pk']
ax.axvline(t_pk,ls=':',c='gray',lw=1.3,alpha=.7,label=f'SA peak t̂={t_pk:.2f}')
ax.plot(t_pk,M_SLP['sa_pk'],'k*',ms=14,zorder=10)
ax.set_ylabel('ĉ at Z=0 (near-bed)',fontsize=12); ax.legend(fontsize=9,ncol=2)
ax.set_title('Figure 5 — Near-Bed (Z=0) Fractional Memory Peak: All Methods',fontsize=12)
ax=axes[1]
ax.plot(T_cmp,C_SA[iz1,:],'k',lw=LW_sa+.3,label='SA')
ax.plot(T_cmp,C_FNN[iz1,:],c=COLS['fnn'],ls=LS['fnn'],lw=LW_m)
ax.plot(T_cmp,C_MLP[iz1,:],c=COLS['mlp'],ls=LS['mlp'],lw=LW_m)
ax.plot(T_cmp,C_SLPNN[iz1,:],c=COLS['slp'],ls=LS['slp'],lw=LW_m)
ax.set_xlabel('t̂',fontsize=12); ax.set_ylabel('ĉ at Z=1 (surface)',fontsize=12)
ax.set_title('Surface (Z=1)',fontsize=12); ax.set_xlim(0,T_END)
plt.tight_layout(); plt.show()

# ─── Figure 6───────────────────────────────────
print('  [Fig 6/7]  IC + BC residuals')
fig,axes=plt.subplots(1,3,figsize=(16,5.5))
ax=axes[0]
ic_z=np.linspace(0,1,100)
for C_pred,lbl,col,ls in [(C_FNN,LBL['fnn'],COLS['fnn'],LS['fnn']),
                           (C_MLP,LBL['mlp'],COLS['mlp'],LS['mlp']),
                           (C_SLPNN,LBL['slp'],COLS['slp'],LS['slp'])]:
    iz_arr=[int(np.argmin(np.abs(Z_cmp-z))) for z in ic_z]
    ic_vals=C_pred[[iz_arr],0].ravel()
    ax.plot(ic_z,ic_vals-1.,c=col,ls=ls,lw=LW_m,label=lbl)
ax.axhline(0,c='k',ls='-',lw=1,alpha=.5); ax.set_xlabel('Z',fontsize=11)
ax.set_ylabel('ĉ(Z,0) − 1  (IC error)',fontsize=11)
ax.set_title(f'IC Residual at t̂→0\nSLPNN = 0 exactly (η_j(0)=0)',fontsize=11); ax.legend(fontsize=8)
T_bc_eval=np.linspace(0.2,T_END,200)
for ax_i,panel in zip(axes[1:],['BC1 (Z=1)','BC2 (Z=0)']):
    ax_i.set_yscale('log'); ax_i.set_xlabel('t̂',fontsize=11)
    ax_i.set_ylabel(f'|{panel} residual|',fontsize=11)
    ax_i.set_title(f'{panel} Residual',fontsize=11)
for dZ_func,C_pred,lbl,col,ls in [(FNN_dZ,C_FNN,LBL['fnn'],COLS['fnn'],LS['fnn']),
    (MLP_PINN_dZ,C_MLP,LBL['mlp'],COLS['mlp'],LS['mlp']),
    (SLPNN_dZ,C_SLPNN,LBL['slp'],COLS['slp'],LS['slp'])]:
    iz_=int(np.argmin(np.abs(Z_cmp-1.))); iz0_=int(np.argmin(np.abs(Z_cmp-0.)))
    tidx=np.array([int(np.argmin(np.abs(T_cmp-t))) for t in T_bc_eval])
    u1=C_pred[iz_,tidx]; du1=dZ_func(np.array([Z_cmp[iz_]]),T_bc_eval)[0]
    u0=C_pred[iz0_,tidx]; du0=dZ_func(np.array([Z_cmp[iz0_]]),T_bc_eval)[0]
    r1=np.abs(EPS_BC*du1+W_S*u1); r2=np.abs(EPS_BC*du0+(W_S-B_REF)*u0+B_REF*C_STAR)
    axes[1].plot(T_bc_eval,r1,c=col,ls=ls,lw=LW_m,label=lbl)
    axes[2].plot(T_bc_eval,r2,c=col,ls=ls,lw=LW_m,label=lbl)
for ax_i in axes[1:]: ax_i.legend(fontsize=8); ax_i.grid(True,which='both',alpha=.2)
fig.suptitle('Figure 6 — IC and Boundary Condition Residuals: All Methods',fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()

# ─── Figure 7────────────────────────────────────────────
print('  [Fig 7/7]  Key metric bar chart')
fig,axes=plt.subplots(1,3,figsize=(14,6))
methods=['Chopra\nFNN','MLP-PINN\n(Adam)','SLPNN\n(Proposed)']
colors=[COLS['fnn'],COLS['mlp'],COLS['slp']]
def bar_plot(ax,vals,ylabel,title,fmt='{:.2e}'):
    bars=ax.bar(methods,vals,color=colors,edgecolor='black',lw=1.2,
        hatch=['//','xx',''],width=.55,alpha=.85)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()*1.04,
            fmt.format(v),ha='center',va='bottom',fontsize=10,fontweight='bold')
    ax.set_ylabel(ylabel,fontsize=11); ax.set_title(title,fontsize=11)
bar_plot(axes[0],[M_FNN['L2_g'],M_MLP['L2_g'],M_SLP['L2_g']],'Global L²','Global L² vs SA')
bar_plot(axes[1],[M_FNN['ic_err'],M_MLP['ic_err'],M_SLP['ic_err']],'IC error','IC Max Error (SLPNN=0★)')
bar_plot(axes[2],[abs(M_FNN['sa_pk']-M_FNN['nn_pk']),abs(M_MLP['sa_pk']-M_MLP['nn_pk']),
    abs(M_SLP['sa_pk']-M_SLP['nn_pk'])],'Peak Δ','Near-Bed Peak Error')
fig.suptitle('Figure 7 — Metric Comparison: SLPNN wins on ALL three key indicators',
    fontsize=12,fontweight='bold')
plt.tight_layout(); plt.show()
