"""Gaussian-Process-based equality-constraint learning.

This is the only GP experiment file kept in this folder.
It keeps the GP + learnable-mean-prior workflow and the GT x Prior matrix
experiment. Redundant sibling scripts were removed to keep one GP entry point.
"""

import os, math
from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------------------- Config ----------------------
CFG = {
    "DIM": 2,
    "N_POS": 70,
    "N_NEG": 0,

    "NEG_R_MIN": 0.3,
    "NEG_R_MAX": 0.8,

    "NEG_FAR_FRAC": 0.3,

    "SEED": 42,

    "EPOCHS": 500,
    "LR": 1e-2,

    "TUNE_HYPERPARAMS": True,
    "TUNE_STEPS": 80,
    "TUNE_LR": 0.02,

    "GRID_LIM": 1.2,
    "GRID_N": 200,

    "USE_GT_SIGN": False,

    # Margin loss
    "NEG_MARGIN_LAMBDA": 1,   # 约束权重
    "NEG_MARGIN_GAMMA": 0.3,    # 要求 |mφ(x^-)| >= gamma
    "NEG_MARGIN_TAU": 0.05,     # softplus 平滑温度

    "NEG_MARGIN_LAMBDA_THETA": 2.0,   # 或 0.5~2 之间调
    "NEG_MARGIN_GAMMA_THETA":  .3,   # 用更“难”的 margin（0.8~2 试）
    "NEG_MARGIN_TAU_THETA":    0.05,

    "ADAPTIVE_LEVEL": not False,
    "ADAPT_Q": 30.0,

    "KERNEL": "matern32",
    "RQ_ALPHA": 1.0,
    "RBF_LENGTHSCALE": 0.3,
    "RBF_AMPLITUDE": 1.0,
    "NOISE_STD": 0.05,

    "EIKONAL_LAMBDA": 0e-3,   # 先从 5e-3 ~ 2e-2 试；不够就上 5e-2
    "EIKONAL_EPS":    0.04,   # 取样离零层更近的两侧
    "EIKONAL_SAMPLES": 256,

    "SHOW_UNCERTAINTY": True,
    "UNCERT_BATCH": 2048,
    "UNCERT_CMAP": "YlOrBr",

    "BAND_MODE": "sigma",   # None | "fixed" | "sigma"
    "BAND_TAU": 0.1,
    "BAND_K": 1.0,
    "BAND_FILL": True,

    "OUT_DIR": "results",
}

# ---------------------- Utils ----------------------
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def unit_norm(v, eps=1e-9):
    return v / (torch.norm(v) + eps)

def project_to_psd(A: torch.Tensor, min_eig: float = 1e-8) -> torch.Tensor:
    A = 0.5 * (A + A.T)
    evals, Q = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=min_eig)
    return (Q * evals) @ Q.T

def stable_cholesky(A: torch.Tensor, max_tries: int = 7,
                    init_jitter: float = 1e-8, growth: float = 10.0):
    """
    Try Cholesky with growing jitter; if still fails, project to PSD and retry.
    Returns (L, jitter_used, used_psd_project)
    """
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    A = 0.5 * (A + A.T)
    jitter = init_jitter
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(A + jitter * I)
            return L, jitter, False
        except Exception:
            jitter *= growth
    # Fallback: PSD projection
    A_psd = project_to_psd(A, min_eig=jitter)
    L = torch.linalg.cholesky(A_psd)
    return L, jitter, True

def dedup_XY(X, y, decimals=6):
    Xr = torch.round(X * (10**decimals)) / (10**decimals)
    Xr_np = Xr.detach().cpu().numpy()
    uniq, inv = np.unique(Xr_np, axis=0, return_inverse=True)
    uniq = torch.tensor(uniq, dtype=X.dtype, device=X.device)
    y_np = y.detach().cpu().numpy()
    y_mean = np.zeros((uniq.shape[0],), dtype=y_np.dtype)
    for k in range(uniq.shape[0]):
        y_mean[k] = y_np[inv == k].mean()
    y_mean = torch.tensor(y_mean, dtype=y.dtype, device=y.device)
    return uniq, y_mean

def eikonal_reg_on_prior(prior: nn.Module, pos: torch.Tensor,
                         eps: float, ns: int) -> torch.Tensor:
    if pos is None or pos.numel() == 0:
        return torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
    idx = torch.randint(0, pos.shape[0], (min(ns, pos.shape[0]),), device=pos.device)
    base = pos[idx]  # (k,2)
    noise = eps * torch.randn_like(base)
    Xp = (base + noise).detach().clone().requires_grad_(True)
    m = prior(Xp)
    if m.ndim == 2: m = m.squeeze(-1)
    g = torch.autograd.grad(m.sum(), Xp, create_graph=True, retain_graph=True)[0]
    grad_norm = g.norm(dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


# ---------------------- Kernels ----------------------
def rbf_kernel(X, Z, ell, sf2):
    Z = Z.to(dtype=X.dtype, device=X.device)
    d = torch.cdist(X, Z)
    kind = str(CFG.get("KERNEL", "rbf")).lower()
    if kind == "rbf":
        K = sf2 * torch.exp(-0.5 * (d**2) / (ell**2))
    elif kind == "matern32":
        r = (math.sqrt(3.0) / ell) * d
        K = sf2 * (1.0 + r) * torch.exp(-r)
    elif kind == "rq":
        alpha = torch.tensor(float(CFG.get("RQ_ALPHA", 1.0)),
                             dtype=X.dtype, device=X.device)
        K = sf2 * torch.pow(1.0 + (d**2) / (2.0 * alpha * (ell**2)), -alpha)
    else:
        raise ValueError(f"Unknown kernel kind: {kind}")
    return K

# ---------------------- GP cache / NLL ----------------------
@torch.no_grad()
def build_gp_cache(X, ell, sf2, sn2):
    K = rbf_kernel(X, X, ell, sf2)
    Ky = K + (sn2 + 1e-10) * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
    Ky = 0.5 * (Ky + Ky.T)
    L, _, _ = stable_cholesky(Ky)
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    return {"L": L, "logdet": logdet}

def gp_marginal_nll_only_phi_cached(X, y, mphi, cache):
    diff = (y.view(-1) - mphi.view(-1)).unsqueeze(-1)
    alpha = torch.cholesky_solve(diff, cache["L"])
    datafit = 0.5 * (diff.T @ alpha).squeeze()
    return datafit + 0.5 * cache["logdet"]

def gp_marginal_nll_full(X, y, mphi, ell, sf2, sn2):
    K = rbf_kernel(X, X, ell, sf2)
    Ky = K + (sn2 + 1e-10) * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
    Ky = 0.5 * (Ky + Ky.T)
    L, _, _ = stable_cholesky(Ky)
    diff = (y.view(-1) - mphi.view(-1)).unsqueeze(-1)
    alpha = torch.cholesky_solve(diff, L)
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    datafit = 0.5 * (diff.T @ alpha).squeeze()
    return datafit + 0.5 * logdet

def _soft_hinge_mean(mu: torch.Tensor, margin: float, tau: float):
    # softplus((γ - |μ|)/τ) * τ
    z = (margin - mu.abs()) / tau
    return torch.log1p(torch.exp(z)) * tau

def tune_hyperparams_ml_ii(
    X, y, prior, ell, sf2, sn2,
    steps=80, lr=0.05, cfg=None,
    neg_np=None,                 # ★ 新增：负样本（numpy）
    hinge_w=0.0, gamma=0.2, tau=0.05  # ★ 新增：间隔超参
):
    log_ell = torch.log(ell.detach().clone()).requires_grad_(True)
    log_sf2 = torch.log(sf2.detach().clone()).requires_grad_(True)
    log_sn2 = torch.log(sn2.detach().clone()).requires_grad_(True)
    opt = optim.Adam([log_ell, log_sf2, log_sn2], lr=lr)

    MIN_ELL = torch.tensor(0.08,  dtype=X.dtype, device=X.device)
    MIN_SN2 = torch.tensor(1e-6,  dtype=X.dtype, device=X.device)
    MIN_SF2 = torch.tensor(1e-6,  dtype=X.dtype, device=X.device)

    # 预备负样本张量
    Xneg = None
    if (neg_np is not None) and (len(neg_np) > 0):
        Xneg = torch.from_numpy(neg_np).to(dtype=X.dtype, device=X.device)

    for _ in range(steps):
        opt.zero_grad()
        ell_t = torch.exp(torch.clamp(log_ell, min=torch.log(MIN_ELL)))
        sf2_t = torch.exp(torch.clamp(log_sf2, min=torch.log(MIN_SF2)))
        sn2_t = torch.exp(torch.clamp(log_sn2, min=torch.log(MIN_SN2)))

        # 训练空间下的先验均值 m_phi
        mphi = prior(X)  # 你现在无 log-warp，UDF 也直接用 prior(X)

        # 边际似然项（和你原来的等价）
        K  = rbf_kernel(X, X, ell_t, sf2_t)
        Ky = K + (sn2_t + 1e-10) * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
        Ky = 0.5 * (Ky + Ky.T)
        L, _, _ = stable_cholesky(Ky)

        diff   = (y.view(-1) - mphi.view(-1)).unsqueeze(-1)
        alpha  = torch.cholesky_solve(diff, L)
        logdet = 2 * torch.sum(torch.log(torch.diag(L)))
        nll    = 0.5 * (diff.transpose(0,1) @ alpha).squeeze() + 0.5 * logdet

        # ★ 负样本后验均值的“间隔”约束：|μθ(x^-)| ≥ γ
        if (hinge_w > 0.0) and (Xneg is not None):
            mneg   = prior(Xneg).unsqueeze(-1)                       # m_phi(x^-)
            Kstar  = rbf_kernel(X, Xneg, ell_t, sf2_t)               # k(X, x^-)
            mu_neg = (mneg + Kstar.transpose(0,1) @ alpha).squeeze(-1)  # posterior mean
            hinge  = _soft_hinge_mean(mu_neg, margin=gamma, tau=tau).mean()
            nll    = nll + hinge_w * hinge

        nll.backward()
        opt.step()

    return torch.exp(log_ell).detach(), torch.exp(log_sf2).detach(), torch.exp(log_sn2).detach()

# ---------------------- Prior helpers ----------------------
def _prior_value_and_grad(prior: nn.Module, X: torch.Tensor):
    Xr = X.detach().clone().requires_grad_(True)
    m = prior(Xr)
    if m.ndim == 2 and m.shape[1] == 1:
        m = m.squeeze(-1)
    if m.requires_grad:
        g = torch.autograd.grad(m.sum(), Xr, create_graph=True, retain_graph=True)[0]
    else:
        g = torch.zeros_like(Xr); m = m.detach()
    return m, g

def make_udf_mean_from_signed_prior(prior: nn.Module, X: torch.Tensor) -> torch.Tensor:
    # 你当前版本：直接用 prior 值（不取绝对值/softabs）
    m_sdf, _ = _prior_value_and_grad(prior, X)
    return m_sdf

# ---------------------- GT constraints ----------------------
def make_gt_constraint(name: str, dim: int, grid_lim: float):
    name = name.lower(); assert dim == 2

    if name == "circle":
        def h(P): x, y = P[:, 0], P[:, 1]; return x*x + y*y - 1
        def grad(P):
            G = np.stack([2*P[:,0], 2*P[:,1]], 1)
            G = G/(np.linalg.norm(G, axis=1, keepdims=True)+1e-9); return G
        def sample_pos(n): t = np.random.rand(n)*2*np.pi; return np.stack([np.cos(t), np.sin(t)], 1)
        def curve(M=1200): t = np.linspace(0,2*np.pi,M); return np.stack([np.cos(t), np.sin(t)],1)
        return h, grad, sample_pos, curve

    if name == "wavy_circle":
        def h(P):
            x, y = P[:,0], P[:,1]
            r = np.sqrt(x**2 + y**2) + 1e-9
            th = np.arctan2(y, x); r_t = 1 + 0.2*np.sin(4*th)
            return r - r_t
        def grad(P):
            x, y = P[:,0], P[:,1]
            r = np.sqrt(x**2 + y**2) + 1e-9
            G = np.stack([x/r, y/r], 1)
            G = G/(np.linalg.norm(G, axis=1, keepdims=True)+1e-9); return G
        def sample_pos(n):
            t = np.random.rand(n)*2*np.pi
            r = 1 + 0.2*np.sin(4*t)
            return np.stack([r*np.cos(t), r*np.sin(t)], 1)
        def curve(M=1200):
            t = np.linspace(0, 2*np.pi, M)
            r = 1 + 0.2*np.sin(4*t)
            return np.stack([r*np.cos(t), r*np.sin(t)], 1)
        return h, grad, sample_pos, curve

    if name == "ellipse":
        a, b = 1.2, 0.7
        def h(P):
            x, y = P[:,0], P[:,1]
            return (x/a)**2 + (y/b)**2 - 1
        def grad(P):
            x, y = P[:,0], P[:,1]
            G = np.stack([2*x/(a*a), 2*y/(b*b)], 1)
            G = G/(np.linalg.norm(G, axis=1, keepdims=True)+1e-9); return G
        def sample_pos(n):
            t = np.random.rand(n)*2*np.pi
            return np.stack([a*np.cos(t), b*np.sin(t)], 1)
        def curve(M=1200):
            t = np.linspace(0,2*np.pi,M)
            return np.stack([a*np.cos(t), b*np.sin(t)], 1)
        return h, grad, sample_pos, curve

    if name == "sine_line":
        e, k = 0.05, 4.0
        def h(P): x, y = P[:,0], P[:,1]; return y - e*np.sin(k*x)
        def grad(P):
            x = P[:,0]
            G = np.stack([-e*k*np.cos(k*x), np.ones_like(x)], 1)
            G = G/(np.linalg.norm(G, axis=1, keepdims=True)+1e-9); return G
        def sample_pos(n):
            xs = np.random.uniform(-0.9*grid_lim, 0.9*grid_lim, size=n)
            ys = e*np.sin(k*xs)
            return np.stack([xs, ys], 1)
        def curve(M=1200):
            xs = np.linspace(-grid_lim, grid_lim, M)
            ys = e*np.sin(k*xs)
            return np.stack([xs, ys], 1)
        return h, grad, sample_pos, curve

    if name == "lemniscate":
        a = 1.0
        def h(P):
            x, y = P[:,0], P[:,1]
            r2 = x*x + y*y
            return r2*r2 - 2*(a*a)*(x*x - y*y)
        def grad(P):
            x, y = P[:,0], P[:,1]
            r2 = x*x + y*y
            gx = 4*x*r2 - 4*(a*a)*x
            gy = 4*y*r2 + 4*(a*a)*y
            G = np.stack([gx, gy], 1)
            G = G/(np.linalg.norm(G, axis=1, keepdims=True)+1e-9); return G

        def _r_and_dr(theta):
            c2 = np.cos(2.0*theta)
            r2 = 2.0*(a*a)*np.maximum(c2, 0.0)
            r = np.sqrt(r2)
            s2 = np.sin(2.0*theta)
            dr = np.where(r>1e-12, -(2.0*a*a)*s2/r, 0.0)
            return r, dr

        def _resample_interval(theta_lo, theta_hi, n):
            pad = 1e-4
            thetas = np.linspace(theta_lo+pad, theta_hi-pad, 3000)
            r, dr = _r_and_dr(thetas)
            w = np.sqrt(r*r + dr*dr) + 1e-12
            s = np.cumsum(w); s = (s - s[0])/(s[-1]-s[0] + 1e-12)
            u = (np.arange(n)+0.5)/n
            th = np.interp(u, s, thetas)
            r, _ = _r_and_dr(th)
            xs, ys = r*np.cos(th), r*np.sin(th)
            return np.stack([xs, ys], 1)

        intervals = [(-np.pi/4, np.pi/4), (3*np.pi/4, 5*np.pi/4)]

        def sample_pos(n):
            def arc_len(lo, hi):
                th = np.linspace(lo+1e-3, hi-1e-3, 2000)
                r, dr = _r_and_dr(th)
                return float(np.trapz(np.sqrt(r*r + dr*dr), th))
            Ls = np.array([arc_len(*itv) for itv in intervals])
            Ls = Ls/(Ls.sum()+1e-12)
            n1 = max(1, int(round(n*Ls[0]))); n2 = max(1, n-n1)
            P1 = _resample_interval(intervals[0][0], intervals[0][1], n1)
            P2 = _resample_interval(intervals[1][0], intervals[1][1], n2)
            P = np.vstack([P1, P2]).astype(np.float32)
            idx = np.random.permutation(len(P))
            return P[idx]

        def curve(M=1600):
            m1 = M//2; m2 = M - m1
            P1 = _resample_interval(intervals[0][0], intervals[0][1], m1)
            P2 = _resample_interval(intervals[1][0], intervals[1][1], m2)
            return np.vstack([P1, P2]).astype(np.float32)

        return h, grad, sample_pos, curve

    raise ValueError("Unknown GT constraint")

# ---------------------- Data ----------------------
def make_negatives_near_pos(pos: np.ndarray, want: int, r_min: float, r_max: float, grid_lim: float):
    if want <= 0 or pos.shape[0] == 0:
        return np.zeros((0,2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    Np = pos.shape[0]
    k = max(1, int(np.ceil(want / Np)))
    pts, radii = [], []
    for i in range(Np):
        theta = np.random.uniform(0.0, 2*np.pi, size=k)
        r = np.random.uniform(r_min, r_max, size=k)
        offset = np.stack([r*np.cos(theta), r*np.sin(theta)], 1)
        cand = pos[i][None, :] + offset
        cand = np.clip(cand, -grid_lim, grid_lim)
        # 微抖动，避免共线/重叠导致核退化
        cand += np.random.normal(0, 1e-6, size=cand.shape).astype(np.float32)
        pts.append(cand.astype(np.float32)); radii.append(r.astype(np.float32))
    neg = np.concatenate(pts, 0); rad = np.concatenate(radii, 0)
    if neg.shape[0] > want:
        idx = np.random.choice(neg.shape[0], want, replace=False)
        neg, rad = neg[idx], rad[idx]
    return neg, rad

def build_dataset(cfg):
    """
    构造数据：
      - 训练 GP 的监督数据：只用正样本 (y = 0)
      - 负样本 neg：仅用于间隔约束（不进入 y）
    """
    h, grad, sample_pos, curve = make_gt_constraint(
        cfg["GT_NAME"], cfg["DIM"], cfg["GRID_LIM"]
    )

    # 正样本：在真实等值线上
    pos = sample_pos(cfg["N_POS"]).astype(np.float32)

    # 负样本：环形带近表面点（只作约束，不回归）
    neg, _ = make_negatives_near_pos(
        pos, cfg["N_NEG"],
        cfg["NEG_R_MIN"], cfg["NEG_R_MAX"],
        cfg["GRID_LIM"]
    )
    neg = neg.astype(np.float32)

    # （可选）远锚点也作为“负约束点”，不进入 y
    if not cfg.get("USE_GT_SIGN", False):
        frac = float(cfg.get("NEG_FAR_FRAC", 0.0))
        n_far = int(frac * cfg["N_NEG"])
        if n_far > 0:
            far = np.random.uniform(
                -cfg["GRID_LIM"], cfg["GRID_LIM"], size=(n_far, 2)
            ).astype(np.float32)
            neg = np.vstack([neg, far])

    # 监督数据：只用正样本，y=0
    X = torch.tensor(pos, dtype=torch.float32)
    y = torch.zeros(len(pos), dtype=torch.float32)

    return X, y, pos, neg, h, grad, curve


# ---------------------- Priors ----------------------
class MeanPriorBase(nn.Module):
    def forward(self, X): raise NotImplementedError

class MeanZero(MeanPriorBase):
    def __init__(self, d): super().__init__(); self.d = d
    def forward(self, X):  return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

class MeanPlane(MeanPriorBase):
    def __init__(self, d):
        super().__init__(); self.n_raw = nn.Parameter(torch.randn(d)); self.b = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        n = unit_norm(self.n_raw)
        return X @ n + self.b

class MeanPoly2D(MeanPriorBase):
    """
    2D 多项式先验：m(x,y) = sum_{i+j<=D} w_{i,j} * x^i * y^j
    - D 为总次数（degree）
    - 系数 w 为可学习参数（φ）
    - 返回形状：(N,)
    说明：
      - 只依赖 X（可求梯度）；不会引入额外状态，兼容你现有训练与可视化逻辑
      - 为稳定起见，对输入做可选的线性预处理（可关闭）
    """
    def __init__(self, degree: int, use_input_norm: bool = True):
        super().__init__()
        assert degree >= 0, "degree must be non-negative"
        self.degree = int(degree)
        self.use_input_norm = bool(use_input_norm)

        # 预生成所有 (i,j) 指数对（i+j<=D）
        exps = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                exps.append((i, j))
        self.register_buffer("exp_i", torch.tensor([e[0] for e in exps], dtype=torch.long))
        self.register_buffer("exp_j", torch.tensor([e[1] for e in exps], dtype=torch.long))
        self.num_terms = len(exps)

        self.coeff = nn.Parameter(10* torch.randn(self.num_terms))

        #（可选）对输入做仿射归一化，帮助数值稳定（可关闭）
        if self.use_input_norm:
            self.c = nn.Parameter(torch.zeros(2))           # 平移中心
            self.log_s = nn.Parameter(torch.zeros(1))       # 等比例缩放（正的）
        else:
            # 放占位，保持属性存在
            self.register_parameter("c", None)
            self.register_parameter("log_s", None)

    def _design(self, X: torch.Tensor) -> torch.Tensor:
        """
        生成设计矩阵 Φ(X) ∈ R^{N×M}，列为各单项式 x^i y^j
        """
        if self.use_input_norm:
            s = torch.exp(self.log_s) + 1e-6
            Xn = (X - self.c) / s
        else:
            Xn = X

        x = Xn[:, 0]
        y = Xn[:, 1]

        # 先把各次幂缓存好，减少重复 pow 调用
        max_i = int(self.exp_i.max().item())
        max_j = int(self.exp_j.max().item())
        # 避免空 pow(·,0) 的边界
        x_pows = [torch.ones_like(x)]
        y_pows = [torch.ones_like(y)]
        for k in range(1, max_i + 1):
            x_pows.append(x_pows[-1] * x)
        for k in range(1, max_j + 1):
            y_pows.append(y_pows[-1] * y)

        # 拼设计矩阵
        cols = []
        for ii, jj in zip(self.exp_i, self.exp_j):
            cols.append(x_pows[int(ii.item())] * y_pows[int(jj.item())])
        Phi = torch.stack(cols, dim=1)   # (N, M)
        return Phi

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Phi = self._design(X)            # (N, M)
        return (Phi @ self.coeff).view(-1)  # (N,)

class MeanRectangle2D(MeanPriorBase):
    """
    Soft-rectangle implicit: softmax_tau( |x'|/a, |y'|/b ) - 1
    x' = R(theta)^T (x - c)
    """
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(2))
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_b = nn.Parameter(torch.zeros(1))
        self.theta = nn.Parameter(torch.zeros(1))
        self.log_tau = nn.Parameter(torch.tensor(-2.0))  # tau≈e^-2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"[MeanRectangle2D] X must be (N,2), got {tuple(X.shape)}")

        # 标量化参数（即使被错误存成多元素，这里也只取第一个）
        a   = torch.exp(self.log_a).reshape(()) + 1e-3
        b   = torch.exp(self.log_b).reshape(()) + 1e-3
        tau = torch.exp(self.log_tau).reshape(()) + 1e-6
        th  = self.theta.reshape(())[()]  # 标量
        ct, st = torch.cos(th), torch.sin(th)

        # 平移 + 旋转（用标量公式，避免 2x2 @ 造成维度问题）
        dx = X[:, 0] - self.c[0]
        dy = X[:, 1] - self.c[1]
        xp =  ct * dx + st * dy     # 相当于 (x - c) @ R 的第一列
        yp = -st * dx + ct * dy     # 相当于 (x - c) @ R 的第二列

        u = torch.abs(xp) / a
        v = torch.abs(yp) / b
        # softmax_tau(u,v) = tau * log( exp(u/tau) + exp(v/tau) )
        m = tau * torch.log(torch.exp(u / tau) + torch.exp(v / tau))
        return m - 1.0

class MeanEllipse2D(MeanPriorBase):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(2))
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_b = nn.Parameter(torch.zeros(1))
        self.theta = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        a = torch.exp(self.log_a) + 1e-3
        b = torch.exp(self.log_b) + 1e-3
        ct, st = torch.cos(self.theta), torch.sin(self.theta)
        row1 = torch.cat((ct, -st), 0); row2 = torch.cat((st, ct), 0)
        R = torch.stack((row1, row2), 0)
        Q = (X - self.c) @ R
        Xp, Yp = Q[:,0], Q[:,1]
        return (Xp/a)**2 + (Yp/b)**2 - 1

class MeanQuadForm(MeanPriorBase):
    def __init__(self, d):
        super().__init__(); self.c = nn.Parameter(torch.zeros(d)); self.L = nn.Parameter(torch.eye(d))
    def forward(self, X):
        V = (X - self.c)
        A = self.L.T @ self.L
        q = (V @ A * V).sum(1)
        return q - 1

class MeanSameAsGT(MeanPriorBase):
    def __init__(self, h_np: Callable[[np.ndarray], np.ndarray]):
        super().__init__(); self.h = h_np
    def forward(self, X):
        with torch.no_grad():
            x = X.detach().cpu().numpy()
            y = self.h(x)
        return torch.tensor(y, dtype=X.dtype, device=X.device)

def make_prior_family(name, dim, h_np=None):
    name = name.lower()
    if name == "zero":       return MeanZero(dim)
    if name == "plane":      return MeanPlane(dim)
    if name == "ellipse2d":  return MeanEllipse2D()
    if name == "quadform":   return MeanQuadForm(dim)
    if name == "same_as_gt": return MeanSameAsGT(h_np)
    if name == "rectangle2d":return MeanRectangle2D()
    if name.startswith("poly"):
        try:
            deg = int(name[4:])
        except Exception:
            raise ValueError(f"Polynomial prior name should be like 'poly3', got '{name}'")
        if dim != 2:
            raise ValueError("MeanPoly2D only supports DIM=2 for now.")
        return MeanPoly2D(degree=deg, use_input_norm=True)

    raise ValueError("Unknown prior family")


def margin_push(value, margin=0.2, tau=0.05):
    # 惩罚 max(0, margin - |value|) 的平滑版；tau 越小越像硬 hinge
    return torch.nn.functional.softplus((margin - value.abs())/tau) * tau

# ---------------------- Training / Predictor ----------------------
def train_only_phi_with_dataset(cfg, prior_name, ds_bundle):
    X, y = ds_bundle["X"], ds_bundle["y"]        # 现在只有正样本 (y==0)
    h, curve = ds_bundle["h"], ds_bundle["curve"]
    neg_np = ds_bundle["neg"]                    # 负样本仅用于约束

    prior = make_prior_family(prior_name, cfg["DIM"], h)
    ell = torch.tensor(cfg["RBF_LENGTHSCALE"], dtype=X.dtype)
    sf2 = torch.tensor(cfg["RBF_AMPLITUDE"]**2, dtype=X.dtype)
    sn2 = torch.tensor(cfg["NOISE_STD"]**2, dtype=X.dtype)

    params = list(prior.parameters())
    opt = optim.Adam(params, lr=cfg["LR"]) if len(params) > 0 else None

    cache = build_gp_cache(X, ell, sf2, sn2)
    losses = []

    def _soft_hinge(vals: torch.Tensor, margin: float, tau: float):
        z = (margin - vals.abs()) / tau
        return torch.log1p(torch.exp(z)) * tau  # softplus

    # ---- φ 阶段训练 ----
    for _ in range(cfg["EPOCHS"]):
        # 1) 基础 NLL（pos 期望 ≈ 0）
        if cfg.get("USE_GT_SIGN", False):
            mphi = prior(X)
        else:
            mphi = make_udf_mean_from_signed_prior(prior, X)
        nll = gp_marginal_nll_only_phi_cached(X, y, mphi, cache)

        # 2) Eikonal 正则（可选）
        reg_w = float(cfg.get("EIKONAL_LAMBDA", 0.0))
        if reg_w > 0:
            pos_t = torch.from_numpy(ds_bundle["pos"]).to(dtype=X.dtype, device=X.device)
            reg = eikonal_reg_on_prior(
                prior, pos_t,
                eps=float(cfg.get("EIKONAL_EPS", 0.02)),
                ns=int(cfg.get("EIKONAL_SAMPLES", 128))
            )
            nll = nll + reg_w * reg

        # 3) 负样本软间隔（可选）
        hinge_w = float(cfg.get("NEG_MARGIN_LAMBDA", 1.0))
        gamma   = float(cfg.get("NEG_MARGIN_GAMMA", 0.2))
        tau     = float(cfg.get("NEG_MARGIN_TAU", 0.05))
        if hinge_w > 0 and neg_np is not None and len(neg_np) > 0:
            Xneg = torch.from_numpy(neg_np).to(dtype=X.dtype, device=X.device)
            if cfg.get("USE_GT_SIGN", False):
                mneg = prior(Xneg)
            else:
                mneg = make_udf_mean_from_signed_prior(prior, Xneg)
            hinge = _soft_hinge(mneg, margin=gamma, tau=tau).mean()
            nll = nll + hinge_w * hinge

        if opt is not None:
            opt.zero_grad(); nll.backward(); opt.step()
        losses.append(float(nll.detach()))

    # ---------- 打印 φ 阶段统计 ----------
    # with torch.no_grad():
    #     pos_t = torch.from_numpy(ds_bundle["pos"]).to(dtype=X.dtype, device=X.device)
    #     mphi_pos = prior(pos_t) if cfg.get("USE_GT_SIGN", False) else make_udf_mean_from_signed_prior(prior, pos_t)
    #     print("\n[φ-phase stats]")
    #     print(f"  mean(mφ(pos)) = {mphi_pos.mean().item():+.6f} | std = {mphi_pos.std().item():.6f}")
    #     if neg_np is not None and len(neg_np) > 0:
    #         neg_t = torch.from_numpy(neg_np).to(dtype=X.dtype, device=X.device)
    #         mphi_neg = prior(neg_t) if cfg.get("USE_GT_SIGN", False) else make_udf_mean_from_signed_prior(prior, neg_t)
    #         abs_mneg = mphi_neg.abs()
    #         print(f"  mean(|mφ(neg)|) = {abs_mneg.mean().item():.6f} | min/max = {abs_mneg.min().item():.6f}/{abs_mneg.max().item():.6f}")

    # ============= θ 阶段（超参数） =============
    if cfg.get("TUNE_HYPERPARAMS", False):
        # 小助手：给定 (ell,sf2,sn2) 计算 μ(P)
        def _gp_mean_for(P, ell_, sf2_, sn2_):
            K  = rbf_kernel(X, X, ell_, sf2_)
            Ky = K + (sn2_ + 1e-10) * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
            L  = torch.linalg.cholesky(Ky)
            mX = prior(X).unsqueeze(-1) if cfg.get("USE_GT_SIGN", False) else make_udf_mean_from_signed_prior(prior, X).unsqueeze(-1)
            alpha = torch.cholesky_solve((y.unsqueeze(-1) - mX), L)
            Kstar = rbf_kernel(X, P, ell_, sf2_)
            mu = (prior(P) if cfg.get("USE_GT_SIGN", False) else make_udf_mean_from_signed_prior(prior, P)).unsqueeze(-1) + Kstar.t() @ alpha
            return mu.squeeze(-1)

        # # —— θ 训练“前”的统计 + 参数值
        # with torch.no_grad():
        #     pos_t = torch.from_numpy(ds_bundle["pos"]).to(dtype=X.dtype, device=X.device)
        #     mu_pos_pre = _gp_mean_for(pos_t, ell, sf2, sn2)
        #     print("\n[θ-phase BEFORE]")
        #     print(f"  ell = {ell.item():.6f} | sf2 = {sf2.item():.6f} | sn2 = {sn2.item():.6f}")
        #     print(f"  mean(μ(pos)) = {mu_pos_pre.mean().item():+.6f} | std = {mu_pos_pre.std().item():.6f}")
        #     if neg_np is not None and len(neg_np) > 0:
        #         neg_t = torch.from_numpy(neg_np).to(dtype=X.dtype, device=X.device)
        #         mu_neg_pre = _gp_mean_for(neg_t, ell, sf2, sn2)
        #         abs_mu_neg_pre = mu_neg_pre.abs()
        #         print(f"  mean(|μ(neg)|) = {abs_mu_neg_pre.mean().item():.6f} | min/max = {abs_mu_neg_pre.min().item():.6f}/{abs_mu_neg_pre.max().item():.6f}")

        # 冻结 φ，仅调 θ
        for p in prior.parameters(): p.requires_grad_(False)

        # θ 阶段的 hinge 配置：若未提供 *_THETA，则回退到训练阶段同名项
        theta_hinge_w = float(cfg.get("NEG_MARGIN_LAMBDA_THETA", cfg.get("NEG_MARGIN_LAMBDA", 1.0)))
        theta_gamma   = float(cfg.get("NEG_MARGIN_GAMMA_THETA",  cfg.get("NEG_MARGIN_GAMMA", 0.2)))
        theta_tau     = float(cfg.get("NEG_MARGIN_TAU_THETA",    cfg.get("NEG_MARGIN_TAU", 0.05)))

        # 调参
        ell, sf2, sn2 = tune_hyperparams_ml_ii(
            X, y, prior, ell, sf2, sn2,
            steps=cfg.get("TUNE_STEPS", 80),
            lr=cfg.get("TUNE_LR", 0.02),
            cfg=cfg,
            neg_np=ds_bundle["neg"],      # 传入负样本（若你在 θ 阶段里也用了 hinge）
            hinge_w=theta_hinge_w, gamma=theta_gamma, tau=theta_tau
        )

        # —— θ 训练“后”的统计 + 参数值
        # with torch.no_grad():
        #     pos_t = torch.from_numpy(ds_bundle["pos"]).to(dtype=X.dtype, device=X.device)
        #     mu_pos_post = _gp_mean_for(pos_t, ell, sf2, sn2)
        #     print("\n[θ-phase AFTER]")
        #     print(f"  ell = {ell.item():.6f} | sf2 = {sf2.item():.6f} | sn2 = {sn2.item():.6f}")
        #     print(f"  mean(μ(pos)) = {mu_pos_post.mean().item():+.6f} | std = {mu_pos_post.std().item():.6f}")
        #     if neg_np is not None and len(neg_np) > 0:
        #         neg_t = torch.from_numpy(neg_np).to(dtype=X.dtype, device=X.device)
        #         mu_neg_post = _gp_mean_for(neg_t, ell, sf2, sn2)
        #         abs_mu_neg_post = mu_neg_post.abs()
        #         print(f"  mean(|μ(neg)|) = {abs_mu_neg_post.mean().item():.6f} | min/max = {abs_mu_neg_post.min().item():.6f}/{abs_mu_neg_post.max().item():.6f}")
        #     print("======================================================\n")

    # 返回
    return prior, (X, y), (ell, sf2, sn2), losses, curve



@torch.no_grad()
def prepare_gp_predictor(X, y, prior, ell, sf2, sn2, cfg):
    K = rbf_kernel(X, X, ell, sf2)
    Ky = K + (sn2 + 1e-10) * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
    Ky = 0.5 * (Ky + Ky.T)
    L, _, _ = stable_cholesky(Ky)

    m_train = prior(X).unsqueeze(-1)

    alpha = torch.cholesky_solve((y.unsqueeze(-1) - m_train), L)

    def pred_fun(P_torch: torch.Tensor) -> torch.Tensor:
        P_torch = P_torch.to(dtype=X.dtype, device=X.device)
        m_grid  = prior(P_torch).unsqueeze(-1)
        Kstar   = rbf_kernel(X, P_torch, ell, sf2)
        mu      = m_grid + Kstar.t() @ alpha
        return mu.squeeze(-1)

    def var_fun(P_torch: torch.Tensor) -> torch.Tensor:
        P_torch = P_torch.to(dtype=X.dtype, device=X.device)
        Kstar   = rbf_kernel(X, P_torch, ell, sf2)
        v       = torch.cholesky_solve(Kstar, L)
        kxx     = torch.full((P_torch.shape[0],), float(sf2), dtype=X.dtype, device=X.device)
        s2      = kxx - (Kstar * v).sum(dim=0)
        return torch.clamp(s2, min=1e-12).sqrt()

    # 可选自适应平移
    if cfg.get("ADAPTIVE_LEVEL", False):
        pos = cfg.get("_POS_FOR_LEVEL", None)
        if pos is not None and len(pos) > 0:
            with torch.no_grad():
                lam = pred_fun(torch.from_numpy(pos).float()).mean().item()
            def shifted_pred(P_torch):
                return pred_fun(P_torch) - lam
            return shifted_pred, var_fun

    return pred_fun, var_fun

# ---------------------- Scoring & Plotting ----------------------
def grid_zero_contour_points_func(f_np, lim: float, n: int, level: float):
    xs = np.linspace(-lim, lim, n); ys = np.linspace(-lim, lim, n)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)
    ZZ = f_np(P).reshape(n, n)
    fig, ax = plt.subplots()
    cs = ax.contour(XX, YY, ZZ, levels=[level])
    plt.close(fig)
    segs = getattr(cs, "allsegs", None)
    if not segs or not segs[0]: return np.zeros((0, 2))
    return np.concatenate(segs[0], 0)

def chamfer(A, B):
    if len(A) == 0 or len(B) == 0: return np.inf
    d2 = ((A[:, None, :] - B[None, :, :])**2).sum(-1)
    return 0.5 * (np.sqrt(d2.min(1)).mean() + np.sqrt(d2.min(0)).mean())

def draw_cell(ax, gt_curve, pred_fun_torch, pos_np, neg_np, level,
              grid_lim=1.8, grid_n=256, show_score=None,
              band_mode=None, band_tau=0.1, band_k=2.0,
              var_fun_torch=None, band_fill=True):
    ax.clear()
    xs = np.linspace(-grid_lim, grid_lim, grid_n)
    ys = np.linspace(-grid_lim, grid_lim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)
    with torch.no_grad():
        MU = pred_fun_torch(torch.from_numpy(P)).cpu().numpy().reshape(grid_n, grid_n)
    ax.contour(XX, YY, MU, levels=[level], colors='#c2185b', linewidths=2)

    if band_mode in ("fixed", "sigma"):
        if band_mode == "fixed":
            levels = [level - band_tau, level + band_tau]
            ax.contour(XX, YY, MU, levels=levels, colors='#37474F', linestyles='--', linewidths=1.5)
            if band_fill:
                ax.contourf(XX, YY, MU, levels=[level-band_tau, level+band_tau],
                            colors=['#607D8B'], alpha=0.15)
        else:
            if var_fun_torch is None:
                raise ValueError("band_mode='sigma' requires var_fun_torch")
            with torch.no_grad():
                SIG = var_fun_torch(torch.from_numpy(P)).cpu().numpy().reshape(grid_n, grid_n)
            F_lo = MU - (level - band_k * SIG)
            F_hi = MU - (level + band_k * SIG)
            ax.contour(XX, YY, F_lo, levels=[0.0], colors='grey', linestyles='--', linewidths=1.)
            ax.contour(XX, YY, F_hi, levels=[0.0], colors='grey', linestyles='--', linewidths=1.)
            if band_fill:
                band_mask = np.sign(F_lo) * np.sign(F_hi) <= 0
                Zfill = np.where(band_mask, 1.0, np.nan)
                ax.imshow(Zfill, extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
                          origin='lower', alpha=0.10, cmap='Greys')

    if gt_curve is not None and len(gt_curve) > 0:
        ax.plot(gt_curve[:, 0], gt_curve[:, 1], 'k--', lw=2)
    if pos_np is not None and len(pos_np) > 0:
        ax.scatter(pos_np[:, 0], pos_np[:, 1], s=16, c='#2e7d32', alpha=0.40)
    if neg_np is not None and len(neg_np) > 0:
        ax.scatter(neg_np[:, 0], neg_np[:, 1], s=7, c='#d32f2f', alpha=0.30)

    ax.set_aspect('equal'); ax.set_xlim([-grid_lim, grid_lim]); ax.set_ylim([-grid_lim, grid_lim])
    ax.set_xticks([]); ax.set_yticks([])
    if show_score is not None:
        ax.text(0.98, 0.02, f"{show_score:.2f}", ha='right', va='bottom',
                transform=ax.transAxes, fontsize=8, color='#444')

def plot_grid_clean(results_matrix, gt_names, prior_names, level, grid_lim=1.8, grid_n=256, title="GT × Prior (clean)", save_path=None):
    R, C = len(gt_names), len(prior_names)
    fig, axes = plt.subplots(R, C, figsize=(3.2*C, 3.3*R))
    if R == 1 and C == 1: axes = np.array([[axes]])
    elif R == 1: axes = axes[None, :]
    elif C == 1: axes = axes[:, None]

    band_mode = CFG.get("BAND_MODE", None)
    band_tau = CFG.get("BAND_TAU", 0.1)
    band_k = CFG.get("BAND_K", 2.0)
    band_fill = CFG.get("BAND_FILL", True)

    for i in range(R):
        for j in range(C):
            cell = results_matrix[i][j]
            ax = axes[i, j]
            draw_cell(ax,
                      gt_curve=cell.get("gt_curve"),
                      pred_fun_torch=cell["pred_fun_torch"],
                      pos_np=cell.get("pos"),
                      neg_np=cell.get("neg"),
                      level=level,
                      grid_lim=grid_lim, grid_n=grid_n,
                      show_score=cell.get("score"),
                      band_mode=band_mode, band_tau=band_tau, band_k=band_k,
                      var_fun_torch=cell.get("var_fun_torch"),
                      band_fill=band_fill)
            if i == 0: ax.set_title(prior_names[j], fontsize=11, pad=6)
            if j == 0: ax.set_ylabel(gt_names[i], fontsize=11, rotation=90, labelpad=8)

    plt.suptitle(title, fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    if save_path is None: save_path = os.path.join(CFG["OUT_DIR"], "GT×Prior_grid.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

import matplotlib.colors as mcolors

def plot_score_heatmap(score_matrix, gt_names, prior_names,
                       title="Chamfer-based Score (higher=better)", save_path=None):
    R, C = len(gt_names), len(prior_names)
    fig, ax = plt.subplots(figsize=(1.8*C + 2, 1.6*R + 2))

    # 更清晰的配色（可改为 "coolwarm", "YlGnBu", "RdYlBu_r" 等）
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    im = ax.imshow(score_matrix, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(C)); ax.set_xticklabels(prior_names)
    ax.set_yticks(np.arange(R)); ax.set_yticklabels(gt_names)
    ax.set_xlabel('Prior'); ax.set_ylabel('Ground Truth'); ax.set_title(title)

    # 根据底色亮度自适应文字颜色
    for i in range(R):
        for j in range(C):
            val = float(score_matrix[i, j])
            # 计算 colormap 下该值对应的 RGB 亮度
            r, g, b, _ = cmap(norm(val))
            luminance = 0.2126*r + 0.7152*g + 0.0722*b
            txt_color = 'black' if luminance > 0.5 else 'white'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=txt_color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Score')

    plt.tight_layout()
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    if save_path is None:
        save_path = os.path.join(CFG["OUT_DIR"], "GT×Prior_score_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# ---------------------- 3D plotting ----------------------
@torch.no_grad()
def visualize_gp_all_3d(cfg, pred_fun_torch, prior, X, y, ell, sf2, sn2,
                        pos_np=None, neg_np=None, gt_curve=None,
                        grid_lim=1.8, grid_n=120, title_prefix="GP 3D"):
    xs = np.linspace(-grid_lim, grid_lim, grid_n)
    ys = np.linspace(-grid_lim, grid_lim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    P_np = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)
    Pt   = torch.from_numpy(P_np).to(dtype=X.dtype, device=X.device)

    # (1) overall
    ZZ_overall = pred_fun_torch(torch.from_numpy(P_np)).cpu().numpy().reshape(grid_n, grid_n)
    fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XX, YY, ZZ_overall, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label='h(x,y) value')
    try: ax.contour(XX, YY, ZZ_overall, levels=[0.0], zdir='z', offset=0.0, colors='magenta', linewidths=2)
    except: pass
    if pos_np is not None and len(pos_np)>0: ax.scatter(pos_np[:,0], pos_np[:,1], np.zeros(len(pos_np)), color='green', s=20, alpha=1, label='pos')
    if neg_np is not None and len(neg_np)>0: ax.scatter(neg_np[:,0], neg_np[:,1], np.zeros(len(neg_np)), color='red', s=20, alpha=0.7, label='neg')
    if gt_curve is not None and len(gt_curve)>0: ax.plot(gt_curve[:,0], gt_curve[:,1], zs=0.0, zdir='z', color='black', lw=1.2, linestyle='--', label='GT h=0')
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("h(x,y)"); ax.view_init(elev=35, azim=50)
    ax.set_title(f"{title_prefix}: overall h(x,y)"); ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.show()

    # Ky, alpha (training space)
    K = rbf_kernel(X, X, ell, sf2)
    Ky = K + (sn2 + 1e-10) * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)
    Ky = 0.5 * (Ky + Ky.T)
    L, _, _ = stable_cholesky(Ky)
    m_train = prior(X)
    alpha = torch.cholesky_solve((y.unsqueeze(-1) - m_train.unsqueeze(-1)), L)

    # mean/resid on grid (training space)
    m_grid = prior(Pt)
    Kstar = rbf_kernel(X, Pt, ell, sf2)
    resid = (Kstar.t() @ alpha).squeeze(-1)

    def _plot(Z_torch, zlabel, title):
        Z = Z_torch.detach().cpu().numpy().reshape(grid_n, grid_n)
        fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(XX, YY, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.90)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=zlabel)
        try: ax.contour(XX, YY, Z, levels=[0.0], zdir='z', offset=0.0, colors='magenta', linewidths=2)
        except: pass
        if pos_np is not None and len(pos_np)>0: ax.scatter(pos_np[:,0], pos_np[:,1], np.zeros(len(pos_np)), color='green', s=14, alpha=0.9, label='pos')
        if neg_np is not None and len(neg_np)>0: ax.scatter(neg_np[:,0], neg_np[:,1], np.zeros(len(neg_np)), color='red', s=14, alpha=0.7, label='neg')
        if gt_curve is not None and len(gt_curve)>0: ax.plot(gt_curve[:,0], gt_curve[:,1], zs=0.0, zdir='z', color='black', lw=1.0, linestyle='--', label='GT h=0')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel(zlabel); ax.view_init(elev=35, azim=50)
        ax.set_title(title); ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout(); plt.show()

    _plot(m_grid, "mean m(x) (training)", f"{title_prefix}: mean m(x)")
    _plot(resid,  "residual K_*^T alpha", f"{title_prefix}: residual")

@torch.no_grad()
def plot_grid_surface3d(results_matrix, gt_names, prior_names, grid_lim=1.2,
                        grid_n_surface=80, stride=4, level=0.0,
                        title="3D surfaces of predicted h(x,y)",
                        save_path=None, show_points=True, show_gt=True,
                        zclip_quantile=None):
    R, C = len(gt_names), len(prior_names)
    xs = np.linspace(-grid_lim, grid_lim, grid_n_surface)
    ys = np.linspace(-grid_lim, grid_lim, grid_n_surface)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)

    ZZ_cache = [[None for _ in range(C)] for _ in range(R)]
    for i in range(R):
        for j in range(C):
            pred_fun = results_matrix[i][j]["pred_fun_torch"]
            ZZ = pred_fun(torch.from_numpy(P)).cpu().numpy().reshape(grid_n_surface, grid_n_surface)
            if zclip_quantile is not None and 0.0 < zclip_quantile < 0.5:
                lo = np.quantile(ZZ, zclip_quantile); hi = np.quantile(ZZ, 1 - zclip_quantile)
                ZZ = np.clip(ZZ, lo, hi)
            ZZ_cache[i][j] = ZZ

    fig = plt.figure(figsize=(3.6*C, 3.6*R))
    for i in range(R):
        for j in range(C):
            ax = fig.add_subplot(R, C, i*C + j + 1, projection='3d')
            ZZ = ZZ_cache[i][j]
            zmin = float(np.nanmin(ZZ)); zmax = float(np.nanmax(ZZ))
            zmin = min(zmin, level); zmax = max(zmax, level)
            if zmin == zmax: zmin -= 1e-6; zmax += 1e-6
            surf = ax.plot_surface(XX, YY, ZZ, cmap='coolwarm',
                                   vmin=zmin, vmax=zmax,
                                   rstride=stride, cstride=stride,
                                   linewidth=0, antialiased=False, alpha=0.92)
            try:
                ax.contour(XX, YY, ZZ, levels=[level], zdir='z', offset=level, colors='magenta', linewidths=1.6)
            except: pass
            cell = results_matrix[i][j]
            if show_points:
                pos = cell.get("pos"); neg = cell.get("neg")
                if pos is not None and len(pos)>0:
                    ax.scatter(pos[:,0], pos[:,1], np.full(len(pos), level), s=20, c='#2e7d32', alpha=1)
                if neg is not None and len(neg)>0:
                    ax.scatter(neg[:,0], neg[:,1], np.full(len(neg), level), s=20, c='#d32f2f', alpha=0.85)
            if show_gt and (cell.get("gt_curve") is not None):
                gt = cell["gt_curve"]
                ax.plot(gt[:,0], gt[:,1], zs=level, zdir='z', color='black', lw=1.0, linestyle='--')
            ax.set_xlim([-grid_lim, grid_lim]); ax.set_ylim([-grid_lim, grid_lim]); ax.set_zlim([zmin, zmax])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if i == 0: ax.set_title(prior_names[j], fontsize=10, pad=4)
            if j == 0: ax.text2D(-0.1, 0.95, gt_names[i], transform=ax.transAxes, fontsize=10)

    plt.suptitle(title, y=0.98)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04, wspace=0.05, hspace=0.08)
    ax.view_init(elev=55, azim=-60)
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    if save_path is None: save_path = os.path.join(CFG["OUT_DIR"], "grid_surface3d_adaptive.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[3D grid] saved to {save_path}")

# ---------------------- Experiment wrappers ----------------------
def build_dataset_for_gt(cfg, gt_name):
    local = dict(cfg); local["GT_NAME"] = gt_name
    X, y, pos, neg, h, grad, curve = build_dataset(local)
    return {"X": X, "y": y, "pos": pos, "neg": neg, "h": h, "grad": grad, "curve": curve, "cfg": local}

def run_single_with_dataset(cfg, prior_name, ds_bundle):
    prior, (X, y), (ell, sf2, sn2), losses, curve = train_only_phi_with_dataset(cfg, prior_name, ds_bundle)
    # 仅用于 predictor/评估的去重（训练已经完成，不影响已学 φ）
    X, y = dedup_XY(X, y, decimals=6)

    cfg["_POS_FOR_LEVEL"] = ds_bundle["pos"]
    pred_fun_torch, var_fun_torch = prepare_gp_predictor(X, y, prior, ell, sf2, sn2, cfg)

    def pred_fun_np(P_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Pt = torch.from_numpy(P_np)
            out = pred_fun_torch(Pt).cpu().numpy()
        return out

    gt_curve = curve(M=1000)
    level = 0.0
    pred_pts = grid_zero_contour_points_func(pred_fun_np, cfg["GRID_LIM"], cfg["GRID_N"], level=level)
    score = np.exp(-chamfer(pred_pts, gt_curve) / 0.06)

    return {
        "pred_fun_torch": pred_fun_torch,
        "var_fun_torch": var_fun_torch,
        "gt_curve": gt_curve,
        "pos": ds_bundle["pos"],
        "neg": ds_bundle["neg"],
        "score": score,
        "level": level,
        "prior": prior, "X": X, "y": y,
        "ell": ell, "sf2": sf2, "sn2": sn2,
    }

def plot_grid_clean_and_scores(cfg, results_matrix, gts, priors):
    level = 0.0
    plot_grid_clean(results_matrix, gts, priors,
                    level=level,
                    grid_lim=cfg["GRID_LIM"], grid_n=cfg["GRID_N"],
                    title="GT × Prior Constraints (Clean)",
                    save_path=os.path.join(cfg["OUT_DIR"], "GT×Prior_grid.png"))
    score_matrix = np.array([[results_matrix[i][j]["score"] for j in range(len(priors))]
                              for i in range(len(gts))], dtype=np.float32)
    plot_score_heatmap(score_matrix, gts, priors,
                       title="Chamfer-based Score (higher=better)",
                       save_path=os.path.join(cfg["OUT_DIR"], "GT×Prior_score_heatmap.png"))
    return score_matrix

def run_grid_and_plot(cfg,
                      gts=("circle", "wavy_circle", "ellipse", "sine_line", "lemniscate"),
                      priors=("plane", "ellipse2d", "rectangle2d", "same_as_gt"),
                      draw_surface_grid=True):
    gts = list(gts); priors = list(priors)
    datasets_per_gt = {gt: build_dataset_for_gt(cfg, gt) for gt in gts}
    R, C = len(gts), len(priors)
    results_matrix = [[None for _ in range(C)] for _ in range(R)]

    for i, gt in enumerate(gts):
        ds_bundle = datasets_per_gt[gt]
        for j, pr in enumerate(priors):
            results_matrix[i][j] = run_single_with_dataset(cfg, pr, ds_bundle)

    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    score_matrix = plot_grid_clean_and_scores(cfg, results_matrix, gts, priors)

    if draw_surface_grid:
        plot_grid_surface3d(
            results_matrix, gts, priors,
            grid_lim=cfg["GRID_LIM"],
            grid_n_surface=140,
            stride=4,
            level=0.0,
            title="3D surfaces of predicted h(x,y)",
            save_path=os.path.join(cfg["OUT_DIR"], "grid_surface3d.png"),
            show_points=True,
            show_gt=True
        )

    return {"results_matrix": results_matrix, "score_matrix": score_matrix,
            "gts": gts, "priors": priors, "cfg": cfg}

def visualize_single(cfg, gt: str, prior: str,
                     grid_n: int = 150,
                     with_components: bool = True,
                     title_suffix: str = ""):
    print(f"[Visualize] GT='{gt}', Prior='{prior}'")
    ds_bundle = build_dataset_for_gt(cfg, gt)
    cell = run_single_with_dataset(cfg, prior, ds_bundle)
    title_main = f"GT: {gt}, Prior: {prior} — 3D surface of h(x,y)"
    if title_suffix: title_main += f" | {title_suffix}"

    visualize_gp_all_3d(
        cfg,
        pred_fun_torch=cell["pred_fun_torch"],
        prior=cell["prior"],
        X=cell["X"], y=cell["y"],
        ell=cell["ell"], sf2=cell["sf2"], sn2=cell["sn2"],
        pos_np=cell["pos"], neg_np=cell["neg"], gt_curve=cell["gt_curve"],
        grid_lim=cfg["GRID_LIM"], grid_n=grid_n,
        title_prefix=f"GT: {gt}, Prior: {prior}"
    )

# ---------------------- Main ----------------------
def main():
    set_seed(CFG["SEED"])
    run_grid_and_plot(CFG,
                      gts=("circle","wavy_circle","ellipse","sine_line","lemniscate"),
                      priors=("poly2", "poly3", "plane","ellipse2d", "rectangle2d"
                                  # ,"same_as_gt"
                              ),
                      draw_surface_grid=True)
    visualize_single(CFG, gt="wavy_circle", prior="ellipse2d", grid_n=240, with_components=True)
    # visualize_single(CFG, gt="sine_line", prior="poly3", grid_n=240, with_components=True)

if __name__ == "__main__":
    main()
