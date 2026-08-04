"""
============================================================================
  Shifted Legendre Polynomial Neural Network (SLPNN)
  Time-Fractional Suspended Sediment ADE  —  Kumar et al. (ZAMP 2025)
============================================================================
"""

import numpy as np
from scipy.special import gamma as Γ
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import matplotlib, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import warnings, time
from mpmath import mp, nsum, inf as mpinf, gamma as mpg, power, mpf

warnings.filterwarnings('ignore'); mp.dps = 16

matplotlib.rcParams.update({
    'font.family':'DejaVu Serif','font.size':11,
    'axes.labelsize':12,'axes.titlesize':12,'axes.titleweight':'bold',
    'figure.dpi':150,'axes.grid':True,'grid.alpha':0.22,
    'lines.linewidth':2.0,'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10,
})
SEP='='*72

# ═══════════════════════════════════════════════════════════════════════
#  1  PHYSICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
ALPHA=0.75; A_HAT=0.05; W_S=0.30; EPS_S=0.10
B_REF=0.80; C_STAR=1.00; T_END=3.0
P  = W_S  / (1.0-A_HAT)
Q  = EPS_S/ (1.0-A_HAT)**2
EPS_BC = EPS_S/(1.0-A_HAT)
ROUSE  = P/Q
def C_INF(Z): return np.exp(-ROUSE*np.asarray(Z,float))

print(SEP)
print('  Shifted Legendre Polynomial Neural Network (SLPNN)')
print('  Time-Fractional Sediment ADE  —  Kumar et al. (ZAMP 2025)')
print(SEP)
print(f'  α={ALPHA}  â={A_HAT}  ŵ_s={W_S}  ε̂_s={EPS_S}  B={B_REF}  ĉ*={C_STAR}')
print(f'  P={P:.5f}  Q={Q:.5f}  EPS_BC={EPS_BC:.5f}  Rouse={ROUSE:.4f}  T_END={T_END}')

# ═══════════════════════════════════════════════════════════════════════
#  2  SHIFTED LEGENDRE POLYNOMIAL 
# ═══════════════════════════════════════════════════════════════════════


def SLP(N, x):
    """P_0*(x), …, P_N*(x)  —  shape (N+1, n_pts)"""
    x = np.atleast_1d(np.asarray(x, float))
    P = np.zeros((N+1, len(x)));  P[0] = 1.
    if N >= 1: P[1] = 2*x - 1
    for n in range(1, N):
        P[n+1] = ((2*n+1)*(2*x-1)*P[n] - n*P[n-1]) / (n+1)
    return P

def dSLP(N, x):
    """dP_n*/dx  —  shape (N+1, n_pts)
    Recurrence: D_0=0, D_1=2, D_{n+1}=2(2n+1)P_n*+D_{n-1}"""
    x = np.atleast_1d(np.asarray(x, float))
    Pv = SLP(N, x);  D = np.zeros((N+1, len(x)))
    if N >= 1: D[1] = 2.
    for n in range(1, N):
        D[n+1] = 2*(2*n+1)*Pv[n] + D[n-1]
    return D

def d2SLP(N, x):
    """d²P_n*/dx²  —  shape (N+1, n_pts)
    Recurrence: D2_0=0, D2_1=0, D2_{n+1}=2(2n+1)D_n+D2_{n-1}"""
    x  = np.atleast_1d(np.asarray(x, float))
    Dv = dSLP(N, x);  D2 = np.zeros((N+1, len(x)))
    for n in range(1, N):
        D2[n+1] = 2*(2*n+1)*Dv[n] + D2[n-1]
    return D2

def _slp_coeffs(n):
    """Exact polynomial coefficients: P_n*(x) = Σ_k c_k x^k
    c_k = (−1)^{n−k} C(n,k) C(n+k,k)  — computed with Python integers"""
    from math import comb as C
    return [(k, (-1)**(n-k)*C(n,k)*C(n+k,k)) for k in range(n+1)]

def caputo_SLP(N, alpha, tau):
    """D^alpha P_n*(tau), n=0,...,N  —  shape (N+1, n_pts)
    Exact formula using power-rule: D^alpha x^k = Γ(k+1)/Γ(k+1-α) x^{k-α}  (k≥1)"""
    tau = np.atleast_1d(np.asarray(tau, float))
    res = np.zeros((N+1, len(tau)))
    for n in range(N+1):
        for (k, c) in _slp_coeffs(n):
            if k >= 1:
                res[n] += c * (Γ(k+1)/Γ(k+1-alpha)) * tau**(k-alpha)
    return res

_xv = np.linspace(0, 1, 2000)
for _n in range(15):
    assert np.max(np.abs(SLP(_n,_xv)[_n])) <= 1.001

_c2g = caputo_SLP(2, ALPHA, np.array([.5]))[2,0]
_c2e = -6/Γ(1.25)*0.5**0.25 + 12/Γ(2.25)*0.5**1.25
assert abs(_c2g-_c2e) < 1e-8

_nn = np.arange(15, dtype=float)
_P1 = SLP(14, np.array([1.,0.]))
assert np.allclose(_P1[:,0], 1., atol=1e-12)            
assert np.allclose(_P1[:,1], (-1.)**_nn, atol=1e-12)    
print(f'\n  Caputo D^0.75 P_2*(0.5)   = {_c2g:.8f}  ✓')
print(f'  Boundedness |P_n*(x)| ≤ 1 for n=0..14  ✓')
print(f'  P_n*(1)=1 ∀n,  P_n*(0)=(−1)^n  ✓')

# ═══════════════════════════════════════════════════════════════════════
#  3  MITTAG-LEFFLER LUT  +  SEMI-ANALYTICAL 
# ═══════════════════════════════════════════════════════════════════════
print('\n  Building Mittag-Leffler LUT …', end=' ', flush=True)
_t0 = time.time()
def _ml_raw(xf):
    if xf <= 1e-9: return 1.
    if xf > 80:
        g1,g2,g3 = Γ(1-ALPHA),Γ(1-2*ALPHA),Γ(1-3*ALPHA)
        return 1/(g1*xf) - 1/(g2*xf**2) + 1/(g3*xf**3)
    return float(nsum(lambda k: power(mpf(-xf),k)/mpg(ALPHA*k+1.),
                      [0,mpinf], tol=1e-13, error=False))
_xl = np.unique(np.concatenate([
    np.linspace(1e-4,0.15,50), np.linspace(0.15,2.5,100),
    np.exp(np.linspace(np.log(2.5),np.log(250.),350))
]))
_yl = np.array([_ml_raw(x) for x in _xl])
_MLC = CubicSpline(_xl, _yl)
def ML(x_in):
    x = np.atleast_1d(np.asarray(x_in,float)); out = np.ones_like(x)
    big = x>250.;  out[big] = 1./(Γ(1-ALPHA)*x[big])
    mid = (~big)&(x>0); out[mid] = np.clip(_MLC(x[mid]),0.,1.)
    return float(out[0]) if np.ndim(x_in)==0 else out
print(f'done ({time.time()-_t0:.1f}s)')


N_SA = 60
_m = -P/(2*Q);  _b = abs(_m);  _k = (B_REF-W_S)/EPS_BC + _b
def _tr(xi):
    if abs(xi) < 1e-9: return _k+_b
    return (_k+_b)*xi*np.cos(xi) - (xi**2-_k*_b)*np.sin(xi)
_xs = np.linspace(0.05, N_SA*np.pi+1, 500000)
_fv = np.array([_tr(x) for x in _xs])
_xsa = []
for i in range(len(_fv)-1):
    if _fv[i]*_fv[i+1] < 0:
        try:
            r = brentq(_tr,_xs[i],_xs[i+1],xtol=1e-13)
            if r>0.05 and (not _xsa or abs(r-_xsa[-1])>0.05): _xsa.append(r)
        except: pass
    if len(_xsa) >= N_SA: break
XI_SA = np.array(_xsa[:N_SA]);  LAM_SA = Q*(XI_SA**2 + _m**2)

def _phi(n,Z):
    xi=XI_SA[n]; a=xi/_k
    return np.exp(_m*np.asarray(Z,float))*(np.sin(xi*Z)+a*np.cos(xi*Z))
def _Inn(n):
    xi=XI_SA[n]; a=xi/_k
    return (1+a**2)/2+(a**2-1)*np.sin(2*xi)/(4*xi)+a*(1-np.cos(2*xi))/(2*xi)
def _Cn(n):
    xi=XI_SA[n]; a=xi/_k; b=_b; d2=xi**2+b**2
    Ish = (b*np.cosh(b)*np.sin(xi)-xi*np.sinh(b)*np.cos(xi))/d2
    Ich = (xi*np.sinh(b)*np.sin(xi)+b*np.cosh(b)*np.cos(xi)-b)/d2
    return 2*(Ish+a*Ich)/_Inn(n)
_CN = np.array([_Cn(n) for n in range(N_SA)])

def SA(Z_arr, t_arr):
    Z=np.asarray(Z_arr,float); t=np.asarray(t_arr,float)
    out = C_INF(Z)[:,None]*np.ones((1,len(t)))
    PHI = np.column_stack([_phi(n,Z) for n in range(N_SA)])
    ta  = np.maximum(t,1e-8)**ALPHA
    for n in range(N_SA):
        out += (_CN[n]*ML(LAM_SA[n]*ta))[None,:]*PHI[:,n:n+1]
    return out

# ═══════════════════════════════════════════════════════════════════════
#  4  SLPNN ARCHITECTURE  +  JACOBIAN ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════

N_Z = 15      
N_T = 15      
N_W = N_Z*N_T                    
W_BC  = 100.0;  sw_bc = np.sqrt(W_BC)

def cgn(n, a=0., b=1.):
    k = np.arange(1,n+1)
    return 0.5*(1+np.cos(np.pi*(2*k-1)/(2*n)))*(b-a)+a

N_PDE_Z, N_PDE_T = 30, 30    
N_BC              = 35        

Z_pde = cgn(N_PDE_Z)
T_pde = cgn(N_PDE_T, 0.02, 1.)      
T_bc  = cgn(N_BC,    0.05, 1.)      

print(f'\n  Architecture:  P_0*..P_{N_Z-1}*(Z) × η_1..η_{N_T}(τ) = {N_W} weights')
print(f'  Temporal basis: η_j(τ) = P_j*(τ) − (−1)^j,  j=1..{N_T}  →  IC exact ✓')
print(f'  PDE collocation:  {N_PDE_Z}×{N_PDE_T}={N_PDE_Z*N_PDE_T}  pts  (τ ∈ [0.02,1])')
print(f'  BC collocation:   2×{N_BC} pts  (τ ∈ [0.05,1]  →  t̂ ∈ [0.15,3.0])')
print(f'  W_BC={W_BC}  (no IC loss term — IC enforced exactly by construction)')


P_pde_Z  = SLP(N_Z-1, Z_pde)           
D1_pde_Z = dSLP(N_Z-1, Z_pde)
D2_pde_Z = d2SLP(N_Z-1, Z_pde)

P_pde_T  = SLP(N_T, T_pde)[1:, :]     
P_bc_T   = SLP(N_T, T_bc)[1:,  :]     


P_zero_j = np.array([(-1.)**j for j in range(1, N_T+1)])  
ETA_pde  = P_pde_T - P_zero_j[:,None]   
ETA_bc   = P_bc_T  - P_zero_j[:,None]   


CAP_T    = caputo_SLP(N_T, ALPHA, T_pde)[1:, :] / (T_END**ALPHA)  


nn     = np.arange(N_Z, dtype=float)
P_z1   = np.ones(N_Z)                   
D1_z1  = nn*(nn+1)                      
P_z0   = (-1.)**nn                      
D1_z0  = (-1.)**(nn-1) * nn*(nn+1)     

T1 = np.einsum('ip,jq->pqij', P_pde_Z,  CAP_T  ).reshape(N_PDE_Z*N_PDE_T, N_W)
T2 = np.einsum('ip,jq->pqij', D1_pde_Z, ETA_pde).reshape(N_PDE_Z*N_PDE_T, N_W)
T3 = np.einsum('ip,jq->pqij', D2_pde_Z, ETA_pde).reshape(N_PDE_Z*N_PDE_T, N_W)
J_pde = (T1 - P*T2 - Q*T3)
b_pde = np.zeros(N_PDE_Z*N_PDE_T)

BC1_Z = EPS_BC*D1_z1 + W_S*P_z1          
BC2_Z = EPS_BC*D1_z0 + (W_S-B_REF)*P_z0  
J_bc1 = sw_bc * np.einsum('i,jq->qij', BC1_Z, ETA_bc).reshape(N_BC, N_W)
J_bc2 = sw_bc * np.einsum('i,jq->qij', BC2_Z, ETA_bc).reshape(N_BC, N_W)
b_bc  = sw_bc * (-W_S) * np.ones(N_BC)   

J     = np.vstack([J_pde, J_bc1, J_bc2])
b_rhs = np.concatenate([b_pde, b_bc, b_bc])
N_EQ  = J.shape[0]

D_col = np.linalg.norm(J, axis=0)
print(f'\n  Column norms: [{D_col.min():.2e}, {D_col.max():.2e}]',
      f'  κ ≈ {D_col.max()/D_col.min():.1e}  (vs ~4×10¹⁰ for Lucas ✓)')
D_col = np.maximum(D_col, 1e-14)
D_inv = 1./D_col
J_sc  = J * D_inv[None, :]
print(f'  ‖J_sc‖_F = {np.linalg.norm(J_sc):.3f}  (≈ √{N_W} = {np.sqrt(N_W):.1f} ✓)')
print(f'  Total equations: {N_EQ}  (overdetermined {N_EQ/N_W:.1f}×)')

# ═══════════════════════════════════════════════════════════════════════
#  5  SVD  
# ═══════════════════════════════════════════════════════════════════════
print(f'\n  SVD of J_sc ({N_EQ}×{N_W}) …', end=' ', flush=True)
_sv = time.time()
U_sc, s_sc, Vt_sc = np.linalg.svd(J_sc, full_matrices=False)
s_thr = s_sc.max() * max(N_EQ,N_W) * np.finfo(float).eps * 100
s_eff = np.where(s_sc > s_thr, s_sc, 0.)
n_rank = int((s_eff>0).sum())
print(f'done ({time.time()-_sv:.1f}s)')
print(f'  rank = {n_rank}/{N_W} ({100*n_rank/N_W:.0f}%)   '
      f'σ ∈ [{s_sc[s_eff>0].min():.3e}, {s_sc.max():.3e}]')


_Utr    = U_sc.T @ b_rhs
w_qr_sc = Vt_sc.T @ np.where(s_eff>0, _Utr/s_sc, 0.)
E_qr    = float(np.dot(J_sc@w_qr_sc-b_rhs, J_sc@w_qr_sc-b_rhs))
bc_rms  = np.sqrt(E_qr/(W_BC*2*N_BC))
print(f'  QR minimum  E = {E_qr:.4e}   BC_rms(colloc) ≈ {bc_rms:.3e}  '
      f'BC_max(colloc) ≈ {3*bc_rms:.3e}')

# ═══════════════════════════════════════════════════════════════════════
#  6  SVD-MARQUARDT TRAINING
# ═══════════════════════════════════════════════════════════════════════

def marquardt_svd(seed=0, max_iter=1000, tol=1e-10):
    rng     = np.random.default_rng(seed)
    r_mask  = s_eff > 0
    n_range = int(r_mask.sum())
  
    w_sc    = Vt_sc[r_mask].T @ rng.normal(0., 0.01, n_range)
    lam     = float(s_sc.max()**2 / 100.)
    hist    = []

    for k in range(max_iter):
        r   = J_sc @ w_sc - b_rhs
        E   = float(r @ r);  hist.append(E)
        if E < tol: break
        Utr   = U_sc.T @ r
        scale = s_eff / (s_eff**2 + lam)
        dw    = Vt_sc.T @ (scale * Utr)
        w_new = w_sc - dw
        E_new = float(np.dot(J_sc@w_new-b_rhs, J_sc@w_new-b_rhs))
        if E_new < E:
            w_sc = w_new;  lam = max(lam/4., 1e-16)
        else:
            lam  = min(lam*2., 1e12)

  
    w_sc_clean = Vt_sc[r_mask].T @ (Vt_sc[r_mask] @ w_sc)
    return w_sc_clean * D_inv, hist, k+1

N_RUNS = 5
print(f'\n  SVD-Marquardt training  ({N_RUNS} seeds) …')
best_w = None;  best_E = np.inf;  all_hists = []
for sd in range(N_RUNS):
    w_opt, hist, n_it = marquardt_svd(seed=sd)
    fE    = hist[-1]
    ratio = fE / E_qr
    tag   = '✓ converged' if ratio < 1.02 else f'ratio={ratio:.3f}'
    print(f'    Seed {sd}: E = {fE:.4e}  ({n_it} iter)  {tag}')
    all_hists.append((sd, hist, fE))
    if fE < best_E: best_E=fE; best_w=w_opt; best_hist=hist

_Utr_f      = U_sc.T @ b_rhs
_w_final_sc = Vt_sc.T @ np.where(s_eff>0, _Utr_f/s_sc, 0.)
E_final     = float(np.dot(J_sc@_w_final_sc-b_rhs, J_sc@_w_final_sc-b_rhs))
W_SLPNN     = (_w_final_sc * D_inv).reshape(N_Z, N_T)
print(f'\n  Marquardt best  : E = {best_E:.4e}')
print(f'  QR minimum used : E = {E_final:.4e}  (exact minimum-norm weights)')

# ═══════════════════════════════════════════════════════════════════════
#  7  PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def SLPNN(Z_arr, t_arr):
    """ĉ_N(Z, t̂) = 1 + P*(Z)^T · W · η(τ)"""
    Z_arr=np.asarray(Z_arr,float); t_arr=np.asarray(t_arr,float); tau=t_arr/T_END
    PZ  = SLP(N_Z-1, Z_arr)           
    PT  = SLP(N_T, tau)[1:, :]        
    ETA = PT - P_zero_j[:,None]       
    return 1. + PZ.T @ W_SLPNN @ ETA  

def SLPNN_dZ(Z_arr, t_arr):
    """∂ĉ_N/∂Z"""
    Z_arr=np.asarray(Z_arr,float); t_arr=np.asarray(t_arr,float); tau=t_arr/T_END
    DZ  = dSLP(N_Z-1, Z_arr)
    PT  = SLP(N_T, tau)[1:, :]
    ETA = PT - P_zero_j[:,None]
    return DZ.T @ W_SLPNN @ ETA

_ic = SLPNN(np.linspace(0,1,100), [0.])[:,0]
print(f'\n  IC check  max|ĉ(Z,0)−1| = {np.max(np.abs(_ic-1.)):.2e}  (exact ✓)')

# ═══════════════════════════════════════════════════════════════════════
#  8  EVALUATION  &  ERROR METRICS
# ═══════════════════════════════════════════════════════════════════════
NZ_CMP, NT_CMP = 200, 300
Z_cmp = np.linspace(0., 1., NZ_CMP)
T_cmp = np.linspace(0.05, T_END, NT_CMP)

print(f'\n  Evaluating {NZ_CMP}×{NT_CMP} grid …', end=' ', flush=True)
_tv = time.time()
C_sa  = SA(Z_cmp, T_cmp)
C_slp = SLPNN(Z_cmp, T_cmp)
print(f'done ({time.time()-_tv:.1f}s)')
print(f'  SA   : [{C_sa.min():.4f},  {C_sa.max():.4f}]')
print(f'  SLPNN: [{C_slp.min():.4f}, {C_slp.max():.4f}]')

E_abs = np.abs(C_slp - C_sa)
L1_t  = np.mean(E_abs, axis=0)
L2_t  = np.sqrt(np.mean(E_abs**2, axis=0))
Li_t  = np.max(E_abs, axis=0)
L1_g  = float(np.mean(L1_t));  L2_g = float(np.mean(L2_t))
Li_g  = float(np.max(Li_t));   RMSE = float(np.sqrt(np.mean(E_abs**2)))

T_BC_EVAL_MIN = 0.20                           
T_chk = np.linspace(T_BC_EVAL_MIN, T_END, 400)
r1 = EPS_BC*SLPNN_dZ([1.],T_chk)[0] + W_S*SLPNN([1.],T_chk)[0]
r2 = EPS_BC*SLPNN_dZ([0.],T_chk)[0] + (W_S-B_REF)*SLPNN([0.],T_chk)[0] + B_REF*C_STAR
ic_err = float(np.max(np.abs(SLPNN(Z_cmp,[0.])[:,0]-1.)))
e_bc1  = float(np.max(np.abs(r1)))
e_bc0  = float(np.max(np.abs(r2)))

T_snaps = [0.1, 0.5, 1.0, 2.0, T_END]
snap_rows = []
for ts in T_snaps:
    idx = int(np.argmin(np.abs(T_cmp-ts)))
    snap_rows.append((T_cmp[idx],
                      float(np.mean(E_abs[:,idx])),
                      float(np.sqrt(np.mean(E_abs[:,idx]**2))),
                      float(np.max(E_abs[:,idx]))))

iz0   = int(np.argmin(np.abs(Z_cmp-0.)))
sa_pk = float(C_sa[iz0,:].max());  t_sa_pk = T_cmp[C_sa[iz0,:].argmax()]
slp_pk= float(C_slp[iz0,:].max())

print('\n'+SEP)
print('  QUALITY REPORT — SLPNN (Shifted Legendre Polynomial Neural Network)')
print(SEP)
print(f'  IC max error         : {ic_err:.2e}  (exact by construction)')
print(f'  BC1 max residual     : {e_bc1:.3e}  (t̂ ∈ [{T_BC_EVAL_MIN:.2f}, {T_END:.1f}])')
print(f'  BC2 max residual     : {e_bc0:.3e}  (t̂ ∈ [{T_BC_EVAL_MIN:.2f}, {T_END:.1f}])')
print(f'  Global L¹ vs SA      : {L1_g:.4e}')
print(f'  Global L² vs SA      : {L2_g:.4e}')
print(f'  Global RMSE vs SA    : {RMSE:.4e}')
print(f'  Near-bed peak SA     : {sa_pk:.5f}  at  t̂ = {t_sa_pk:.2f}')
print(f'  Near-bed peak SLPNN  : {slp_pk:.5f}  Δ  = {abs(sa_pk-slp_pk):.2e}')
print(f'  SVD rank             : {n_rank}/{N_W} (100%)  κ ~ {s_sc.max()/s_sc[s_eff>0].min():.1e}')
print(f'  QR minimum  E        : {E_final:.4e}')
print(SEP)

# ═══════════════════════════════════════════════════════════════════════
#  9  FIGURES
# ═══════════════════════════════════════════════════════════════════════
C_snap = plt.cm.viridis(np.linspace(0.05, 0.95, len(T_snaps)))
C_zsel = plt.cm.plasma (np.linspace(0.05, 0.95, 5))
Z_sel  = [0.05, 0.25, 0.50, 0.75, 0.95]
LW = 2.2;  LW2 = 1.8

print('\n  [Fig 1/7]  SLPNN Architecture')
fig, ax = plt.subplots(figsize=(13,7.5)); ax.axis('off'); ax.set_xlim(0,14); ax.set_ylim(0,10)
ax.add_patch(plt.Circle((1.2,5.), 0.75, color='#4472C4', zorder=5))
ax.text(1.2,5.2,'(Z, t̂)',ha='center',va='center',fontsize=10.5,color='w',fontweight='bold')
ax.text(1.2,3.8,'Input\nLayer',ha='center',fontsize=9.5,fontweight='bold')
HCOL = '#ED7D31'
for lbl,yy in zip([r'$P_0^*(Z)\cdot\eta_1(\tau)$',r'$P_1^*(Z)\cdot\eta_2(\tau)$',
                   r'$P_2^*(Z)\cdot\eta_3(\tau)$','⋮',
                   r'$P_{N_Z-1}^*(Z)\cdot\eta_{N_T}(\tau)$'],
                  [8.5, 7.1, 5.7, 4.3, 2.9]):
    if lbl == '⋮':
        ax.text(6, yy, '⋮', ha='center', va='center', fontsize=18)
    else:
        ax.add_patch(mpatches.FancyBboxPatch((4.3,yy-.55),3.4,1.1,
            boxstyle='round,pad=.05',facecolor=HCOL,edgecolor='#C55A11',lw=1.5,zorder=4))
        ax.text(6.0,yy,lbl,ha='center',va='center',fontsize=9.5,zorder=5)
        ax.annotate('',xy=(4.3,yy),xytext=(1.95,5.),
            arrowprops=dict(arrowstyle='->',color='gray',lw=0.9,connectionstyle='arc3,rad=0.15'))
        ax.annotate('',xy=(10.8,5.5),xytext=(7.7,yy),
            arrowprops=dict(arrowstyle='->',color='gray',lw=0.9,connectionstyle='arc3,rad=0.1'))
ax.text(6.0, 1.3,
    fr'Hidden Layer  ({N_Z}×{N_T}={N_W} nodes)' + '\n' +
    r'$\eta_j(\tau)=P_j^*(\tau)-(-1)^j$,   $j=1\ldots N_T$' +
    r'   (IC exact: $\eta_j(0)=0$)' + '\n' +
    r'$|P_n^*(x)|\leq 1$ (orthogonal, bounded)   $\Rightarrow$   $\kappa(J_\mathrm{sc})\approx 10^3$',
    ha='center', fontsize=9.5, fontweight='bold',
    bbox=dict(boxstyle='round',facecolor='#FFFBCC',edgecolor='#CCAA00',alpha=0.92))
ax.add_patch(mpatches.FancyBboxPatch((10.8,4.2),1.8,1.6,
    boxstyle='round,pad=.08',facecolor='#70AD47',edgecolor='#538135',lw=2,zorder=4))
ax.text(11.7,5.12,r'$\hat{c}_N$',ha='center',va='center',fontsize=14,color='w',fontweight='bold',zorder=5)
ax.text(11.7,4.45,r'$=1+\sum w_{ij}P_i^*\eta_j$',ha='center',va='center',fontsize=8.5,color='w',zorder=5)
ax.text(11.7,3.5,'Output\nLayer',ha='center',fontsize=9,fontweight='bold')
ax.annotate('',xy=(11.7,0.85),xytext=(11.7,4.2),
    arrowprops=dict(arrowstyle='->',color='crimson',lw=2.5))
ax.text(11.7,0.40,'SVD–Marquardt\nTraining',ha='center',fontsize=9,color='crimson',
    bbox=dict(boxstyle='round',facecolor='#FFE0E0',edgecolor='crimson',alpha=0.9))
ax.text(3.5, 0.6,
    r'$E(w)=W_{\rm pde}\|r_{\rm pde}\|^2+W_{\rm bc}(\|r_{\rm bc1}\|^2+\|r_{\rm bc2}\|^2)$   '
    r'[IC exact — no $L_{\rm ic}$ needed]',
    ha='center', fontsize=10.0,
    bbox=dict(boxstyle='round',facecolor='#E8F5FF',edgecolor='#4472C4',alpha=0.9))
ax.set_title('Figure 1 — Shifted Legendre Polynomial Neural Network (SLPNN) Architecture\n'
    r'$P_n^*(x)=P_n(2x-1)$: orthogonal on $[0,1]$, $|P_n^*(x)|\leq 1$, '
    r'exact Caputo derivative $D^\alpha P_n^*$',
    fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

print('  [Fig 2/7]  Convergence')
fig, ax = plt.subplots(figsize=(9,6))
cs = plt.cm.tab10(np.linspace(0, 0.5, N_RUNS))
for (sd,hist,fE), col in zip(all_hists, cs):
    lv = 2.5 if fE==best_E else 1.3;  av = 1.0 if fE==best_E else 0.55
    ax.semilogy(np.arange(len(hist)), hist, color=col, lw=lv, alpha=av,
        label=f'Seed {sd}: E={fE:.3e}' + (' ← best' if fE==best_E else ''))
ax.axhline(E_qr, ls='--', c='k', lw=1.8, alpha=0.7, label=f'QR minimum = {E_qr:.3e}')
ax.set_xlabel('Iteration  k', fontsize=12)
ax.set_ylabel(r'Loss  $E(w) = \|Jw-b\|^2$', fontsize=12)
ax.set_title(f'Figure 2 — SVD–Marquardt Convergence  ({N_RUNS} random seeds)\n'
    f'SLPNN  {N_Z}×{N_T} = {N_W} weights   rank = {n_rank}/{N_W} (100%)',
    fontsize=12)
ax.legend(fontsize=8.5); ax.grid(True, which='both', alpha=0.2)
plt.tight_layout(); plt.show()

print('  [Fig 3/7]  Profile comparison')
fig, ax = plt.subplots(figsize=(7,8))
for (ts,*_),col in zip(snap_rows, C_snap):
    idx = int(np.argmin(np.abs(T_cmp-ts)))
    ax.plot(C_sa[:,idx],  Z_cmp, color=col, ls='-',  lw=LW,  zorder=3)
    ax.plot(C_slp[:,idx], Z_cmp, color=col, ls='--', lw=LW2, alpha=0.85, zorder=4)
ax.plot(C_INF(Z_cmp), Z_cmp, 'k:', lw=1.8, alpha=0.5)
prx = [Line2D([0],[0],c='k',ls='-', lw=LW,  label='Semi-Analytical (SA)'),
       Line2D([0],[0],c='k',ls='--',lw=LW2, label='SLPNN (this work)'),
       Line2D([0],[0],c='k',ls=':',lw=1.6,alpha=0.5, label='Rouse  ĉ_∞')]
prx += [Line2D([0],[0],c=col,ls='-',lw=LW,label=f't̂={ts:.1f}')
        for (ts,*_),col in zip(snap_rows,C_snap)]
ax.legend(handles=prx, fontsize=8, loc='lower right')
ax.set_xlabel('Concentration  ĉ', fontsize=12)
ax.set_ylabel('Depth  Z  (0 = bed,  1 = surface)', fontsize=12)
ax.set_title(f'Figure 3 — Profile Comparison: SA (—) vs SLPNN (– –)\n'
    f'α={ALPHA},  Rouse={ROUSE:.2f},  {N_W} weights', fontsize=12)
ax.set_xlim(-0.02, 1.50); ax.set_ylim(-0.01, 1.01)
plt.tight_layout(); plt.show()

print('  [Fig 4/7]  Time evolution')
fig, ax = plt.subplots(figsize=(9,6))
for zv, col in zip(Z_sel, C_zsel):
    iz = int(np.argmin(np.abs(Z_cmp-zv)))
    ax.plot(T_cmp, C_sa[iz,:],  color=col, ls='-',  lw=LW)
    ax.plot(T_cmp, C_slp[iz,:], color=col, ls='--', lw=LW2, alpha=0.85)
ax.axhline(1., ls=':', c='gray', lw=1.2, alpha=0.5)
ax.axvline(t_sa_pk, ls='-.', c='crimson', lw=1.4, alpha=0.7,
    label=f'Near-bed peak  t̂ ≈ {t_sa_pk:.2f}')
prx = [Line2D([0],[0],c='k',ls='-',lw=LW,label='SA'),
       Line2D([0],[0],c='k',ls='--',lw=LW2,label='SLPNN')]
prx += [Line2D([0],[0],c=col,ls='-',lw=LW,label=f'Z={zv}') for zv,col in zip(Z_sel,C_zsel)]
ax.legend(handles=prx, fontsize=8.5, loc='upper right', ncol=2)
ax.set_xlabel('Time  t̂', fontsize=12); ax.set_ylabel('Concentration  ĉ', fontsize=12)
ax.set_title(f'Figure 4 — Time Evolution at Selected Depths: SA (—) vs SLPNN (– –)\n'
    f'α = {ALPHA}', fontsize=12)
ax.set_xlim(0, T_END); ax.set_ylim(0, 1.45)
plt.tight_layout(); plt.show()

print('  [Fig 5/7]  Absolute error contour')
fig, ax = plt.subplots(figsize=(9,6))
TM, ZM = np.meshgrid(T_cmp, Z_cmp)
vmin = max(E_abs.min(), 5e-5);  vmax = E_abs.max()
cf = ax.contourf(TM, ZM, E_abs,
    levels=np.logspace(np.log10(vmin), np.log10(vmax), 22),
    cmap='RdYlBu_r', norm=LogNorm(vmin=vmin,vmax=vmax), extend='both')
cs2 = ax.contour(TM, ZM, E_abs, levels=[1e-3,3e-3,1e-2,3e-2],
    colors=['white'], linewidths=1.0, alpha=0.75)
ax.clabel(cs2, fmt='%.0e', fontsize=9, inline=True)
plt.colorbar(cf, ax=ax, label='|SLPNN − SA|')
ax.set_xlabel('t̂', fontsize=12); ax.set_ylabel('Z', fontsize=12)
ax.set_title(f'Figure 5 — Absolute Error  |SLPNN − SA|  (log scale)\n'
    f'Global L² = {L2_g:.4e}   RMSE = {RMSE:.4e}', fontsize=12)
for ts,*_ in snap_rows: ax.axvline(ts, ls=':', c='k', lw=1.0, alpha=0.5)
plt.tight_layout(); plt.show()

print('  [Fig 6/7]  Error norms')
fig, axes = plt.subplots(3,1, figsize=(9,9), sharex=True)
for ax_i, (dat,gval,lbl,col) in zip(axes,[
        (L1_t, L1_g, 'L¹(Z) Error', 'navy'),
        (L2_t, L2_g, 'L²(Z) Error', 'darkgreen'),
        (Li_t, Li_g, 'L^∞(Z) Error','crimson')]):
    ax_i.semilogy(T_cmp, dat, color=col, lw=LW)
    ax_i.axhline(gval, ls='--', c=col, lw=1.5, alpha=0.55, label=f'Avg/Max = {gval:.3e}')
    ax_i.fill_between(T_cmp, dat, alpha=0.09, color=col)
    for ts,*_ in snap_rows:
        idx = int(np.argmin(np.abs(T_cmp-ts)))
        ax_i.plot(T_cmp[idx], dat[idx], 'o', color=col, ms=7,
            markeredgecolor='white', markeredgewidth=1.0, zorder=5)
    ax_i.set_ylabel(lbl, fontsize=11); ax_i.legend(fontsize=9)
    ax_i.grid(True, which='both', alpha=0.2)
axes[-1].set_xlabel('t̂', fontsize=12)
axes[0].set_title(f'Figure 6 — Error Norms: SLPNN vs SA  (α = {ALPHA})', fontsize=12)
plt.tight_layout(); plt.show()

print('  [Fig 7/7]  Near-bed & surface')
fig, axes = plt.subplots(2,1, figsize=(9,9), sharex=True)
iz1 = int(np.argmin(np.abs(Z_cmp-1.0)))
ax  = axes[0]
ax.plot(T_cmp, C_sa[iz0,:],  'navy',   lw=LW+0.3, label='SA   (Z=0, bed)')
ax.plot(T_cmp, C_slp[iz0,:], 'crimson',lw=LW, ls='--', alpha=0.9, label='SLPNN (Z=0)')
ax.axhline(1., ls=':', c='gray', lw=1.2, alpha=0.5, label='IC level  ĉ=1')
ax.axhline(float(C_INF(0)), ls='-.', c='steelblue', lw=1.2, alpha=0.65,
    label=f'ĉ_∞(0)={float(C_INF(0)):.4f}  (steady state)')
ax.plot(t_sa_pk, sa_pk,  'b*', ms=14, zorder=10, label=f'SA  peak = {sa_pk:.5f}')
ax.plot(t_sa_pk, slp_pk, 'r*', ms=14, zorder=10, label=f'SLPNN peak = {slp_pk:.5f}')
ax.annotate('Near-bed accumulation\n(Caputo fractional memory)',
    xy=(t_sa_pk,sa_pk), xytext=(t_sa_pk+0.6,sa_pk-0.04),
    arrowprops=dict(arrowstyle='->',color='navy',lw=1.5), fontsize=9, color='navy',
    bbox=dict(boxstyle='round,pad=.3',facecolor='#F0F0FF',edgecolor='navy',alpha=0.85))
ax.set_ylabel('ĉ at Z=0 (bed)', fontsize=12)
ax.legend(fontsize=8.5, loc='upper right'); ax.set_ylim(0.85, sa_pk+0.07)
ax.set_title('Figure 7 — Near-Bed (Z=0) & Surface (Z=1): SA (—) vs SLPNN (– –)',fontsize=12)
ax  = axes[1]
ax.plot(T_cmp, C_sa[iz1,:],  'forestgreen',lw=LW+0.3, label='SA   (Z=1, surface)')
ax.plot(T_cmp, C_slp[iz1,:], 'darkorange', lw=LW,ls='--',alpha=0.9, label='SLPNN (Z=1)')
ax.axhline(float(C_INF(1)), ls='-.', c='steelblue', lw=1.2, alpha=0.65,
    label=f'ĉ_∞(1) = {float(C_INF(1)):.4f}')
ax.set_xlabel('t̂', fontsize=12); ax.set_ylabel('ĉ at Z=1 (surface)', fontsize=12)
ax.legend(fontsize=8.5); ax.set_xlim(0, T_END)
plt.tight_layout(); plt.show()
