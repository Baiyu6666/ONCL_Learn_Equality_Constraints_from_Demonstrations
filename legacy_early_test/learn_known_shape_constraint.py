import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================
# 小工具：多层感知机（可自定义隐藏层）
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=(64,64), out_dim=3, activation=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# =========================
# 三个形状的 ConstraintNet
# forward(x) -> f(x)  (N,1),  f>=0 可行
# =========================
class CircleConstraintNet(nn.Module):
    # 输出 [cx, cy, rho], r^2 = softplus(rho)
    def __init__(self, hidden=(64,64), init=(0.0, 0.0, 1.0)):
        super().__init__()
        self.head = MLP(1, hidden, 3)
        self.softplus = nn.Softplus()
        self.register_buffer("const_in", torch.ones(1,1))
        # init：把最后一层 bias 设到 (cx,cy,rho0)
        with torch.no_grad():
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
            cx0, cy0, r0 = init
            rho0 = np.log(np.exp(r0**2) - 1.0)
            last = None
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    last = m
            last.bias.copy_(torch.tensor([cx0, cy0, rho0], dtype=torch.float32))

    def params_numpy(self):
        with torch.no_grad():
            out = self.head(self.const_in).squeeze(0)  # (3,)
            cx, cy, rho = out[0].item(), out[1].item(), out[2].item()
            r = float(np.sqrt(self.softplus(torch.tensor(rho)).item()))
        return (cx, cy, r)

    def forward(self, x):
        theta = self.head(self.const_in).squeeze(0)
        cx, cy, rho = theta[0], theta[1], theta[2]
        s = self.softplus(rho)  # r^2
        diff = x - torch.stack([cx,cy])[None,:]
        sq = torch.sum(diff*diff, dim=1, keepdim=True)
        return sq - s

class EllipseConstraintNet(nn.Module):
    # 输出 [cx, cy, a2_raw, b2_raw, theta_raw]
    # a^2 = softplus(a2_raw), b^2 = softplus(b2_raw), theta = angle_limit * tanh(theta_raw)
    def __init__(self, hidden=(64,64), init=(0.0, 0.0, 1.2, 0.8, 0.0), angle_limit=np.pi):
        super().__init__()
        self.head = MLP(1, hidden, 5)
        self.softplus = nn.Softplus()
        self.angle_limit = angle_limit
        self.register_buffer("const_in", torch.ones(1,1))
        with torch.no_grad():
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
            cx0, cy0, a0, b0, th0 = init
            a2_0 = np.log(np.exp(a0*a0) - 1.0)
            b2_0 = np.log(np.exp(b0*b0) - 1.0)
            last = None
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    last = m
            last.bias.copy_(torch.tensor([cx0, cy0, a2_0, b2_0, np.arctanh(np.clip(th0/self.angle_limit, -0.999, 0.999))], dtype=torch.float32))

    def params_numpy(self):
        with torch.no_grad():
            out = self.head(self.const_in).squeeze(0)
            cx, cy, a2_raw, b2_raw, t_raw = [x.item() for x in out]
            a = float(np.sqrt(self.softplus(torch.tensor(a2_raw)).item()))
            b = float(np.sqrt(self.softplus(torch.tensor(b2_raw)).item()))
            theta = float(self.angle_limit * np.tanh(t_raw))
        return (cx, cy, a, b, np.degrees(theta))

    def forward(self, x):
        theta = self.head(self.const_in).squeeze(0)
        cx, cy, a2_raw, b2_raw, t_raw = theta
        a2 = self.softplus(a2_raw)
        b2 = self.softplus(b2_raw)
        th = self.angle_limit * torch.tanh(t_raw)
        c, s = torch.cos(th), torch.sin(th)
        R_T = torch.stack([torch.stack([ c, s]),
                           torch.stack([-s, c])])  # 2x2 (R^T)
        diff = x - torch.stack([cx,cy])[None,:]              # (N,2)
        y = diff @ R_T.T                                     # (N,2)
        val = (y[:,0:1]**2)/a2 + (y[:,1:2]**2)/b2           # (N,1)
        return val - 1.0

class RectangleConstraintNet(nn.Module):
    # 输出 [cx, cy, w_raw, h_raw, theta_raw]
    # w = softplus(w_raw), h = softplus(h_raw), theta = angle_limit * tanh(theta_raw)
    # f(x) = max(|x'| - w, |y'| - h)，其中 x' = R^T (x-c)
    def __init__(self, hidden=(64,64), init=(0.0, 0.0, 1.2, 0.7, 0.0), angle_limit=np.pi):
        super().__init__()
        self.head = MLP(1, hidden, 5)
        self.softplus = nn.Softplus()
        self.angle_limit = angle_limit
        self.register_buffer("const_in", torch.ones(1,1))
        with torch.no_grad():
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
            cx0, cy0, w0, h0, th0 = init
            w_raw0 = np.log(np.exp(w0) - 1.0)
            h_raw0 = np.log(np.exp(h0) - 1.0)
            last = None
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    last = m
            last.bias.copy_(torch.tensor([cx0, cy0, w_raw0, h_raw0, np.arctanh(np.clip(th0/self.angle_limit, -0.999, 0.999))], dtype=torch.float32))

    def params_numpy(self):
        with torch.no_grad():
            out = self.head(self.const_in).squeeze(0)
            cx, cy, w_raw, h_raw, t_raw = [x.item() for x in out]
            w = float(self.softplus(torch.tensor(w_raw)).item())
            h = float(self.softplus(torch.tensor(h_raw)).item())
            theta = float(self.angle_limit * np.tanh(t_raw))
        return (cx, cy, w, h, np.degrees(theta))

    def forward(self, x):
        theta = self.head(self.const_in).squeeze(0)
        cx, cy, w_raw, h_raw, t_raw = theta
        w = self.softplus(w_raw)
        h = self.softplus(h_raw)
        th = self.angle_limit * torch.tanh(t_raw)
        c, s = torch.cos(th), torch.sin(th)
        R_T = torch.stack([torch.stack([ c, s]),
                           torch.stack([-s, c])])  # 2x2 (R^T)
        diff = x - torch.stack([cx,cy])[None,:]     # (N,2)
        y = diff @ R_T.T
        # 近似 signed distance（L∞ 范式风格）
        g1 = torch.abs(y[:,0:1]) - w
        g2 = torch.abs(y[:,1:2]) - h
        return torch.maximum(g1, g2)   # >=0 在外部
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle

# ===================================================
# 可视化
# ===================================================
def draw_circle(ax, c, r, **kw):
    circ = Circle(c, r, fill=False, **kw)
    ax.add_patch(circ)

def draw_ellipse(ax, c, a, b, theta_deg, **kw):
    ell = Ellipse(xy=c, width=2*a, height=2*b, angle=theta_deg, fill=False, **kw)
    ax.add_patch(ell)

def draw_rectangle(ax, c, w, h, theta_deg, **kw):
    cx, cy = c
    th = np.deg2rad(theta_deg)
    cth, sth = np.cos(th), np.sin(th)
    R = np.array([[cth, -sth],
                  [sth,  cth]], dtype=float)
    corners_local = np.array([
        [ +w, +h],
        [ -w, +h],
        [ -w, -h],
        [ +w, -h],
    ], dtype=float)
    corners_world = corners_local @ R.T + np.array([cx, cy])
    poly = Polygon(corners_world, closed=True, fill=False, **kw)
    ax.add_patch(poly)

def plot_dataset_and_shape(X_pos, X_neg,
                           shape, gt_params, est_params,
                           title="Learned constraint vs. ground-truth"):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if len(X_pos) > 0:
        ax.scatter(X_pos[:,0], X_pos[:,1], s=8, alpha=0.7, label="positive (feasible)")
    if len(X_neg) > 0:
        ax.scatter(X_neg[:,0], X_neg[:,1], s=8, alpha=0.7, label="negative (infeasible)")

    # --- 画 GT ---
    if shape == "circle":
        (cx,cy,r_true) = gt_params
        draw_circle(ax, (cx,cy), r_true, linewidth=2.0, linestyle="--", label="GT circle")
    elif shape == "ellipse":
        (cx,cy,a_true,b_true,theta_true_deg) = gt_params
        draw_ellipse(ax, (cx,cy), a_true, b_true, theta_true_deg, linewidth=2.0, linestyle="--", label="GT ellipse")
    elif shape == "rectangle":
        (cx,cy,w_true,h_true,theta_true_deg) = gt_params
        draw_rectangle(ax, (cx,cy), w_true, h_true, theta_true_deg, linewidth=2.0, linestyle="--", label="GT rectangle")
    elif shape == "union":
        # union 的 gt_params 可以是描述文字或 None，这里不画 GT
        pass

    # --- 画 Learned ---
    if isinstance(est_params, list):  # 多约束组合
        colors = ["b","m","c","y","k"]
        for i,p in enumerate(est_params):
            typ = p[0]
            if typ == "circle":
                _,cx,cy,r_est = p
                draw_circle(ax,(cx,cy),r_est,linewidth=2.2,color=colors[i%len(colors)],label=f"learned circle {i}")
            elif typ == "ellipse":
                _,cx,cy,a_est,b_est,th_est = p
                draw_ellipse(ax,(cx,cy),a_est,b_est,th_est,linewidth=2.2,color=colors[i%len(colors)],label=f"learned ellipse {i}")
            elif typ == "rectangle":
                _,cx,cy,w_est,h_est,th_est = p
                draw_rectangle(ax,(cx,cy),w_est,h_est,th_est,linewidth=2.2,color=colors[i%len(colors)],label=f"learned rect {i}")
    else:
        if shape == "circle":
            (cx,cy,r_est) = est_params
            draw_circle(ax,(cx,cy),r_est,linewidth=2.2,linestyle="-",label="learned circle")
        elif shape == "ellipse":
            (cx,cy,a_est,b_est,theta_est_deg) = est_params
            draw_ellipse(ax,(cx,cy),a_est,b_est,theta_est_deg,linewidth=2.2,linestyle="-",label="learned ellipse")
        elif shape == "rectangle":
            (cx,cy,w_est,h_est,theta_est_deg) = est_params
            draw_rectangle(ax,(cx,cy),w_est,h_est,theta_est_deg,linewidth=2.2,linestyle="-",label="learned rect")

    ax.set_aspect("equal","box")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()

# ===================================================
# 数据生成（加噪声：正负样本都加）
# 说明：加完噪声后不改标签 —— 会产生合理的“标签噪声”
# ===================================================
def _add_noise(x, noise_std, rng):
    if noise_std <= 0: return x
    return x + noise_std * rng.randn(*x.shape)

def sample_circle(center=(0.0,0.0), r=1.2, N_pos=1000, N_neg=1000, seed=0, noise_std=0.3):
    rng = np.random.RandomState(seed)
    cx,cy = center
    def inside(x):
        return np.linalg.norm(x - np.array([cx,cy]), axis=1) <= r
    pos,neg=[],[]
    while len(pos)<N_pos:
        x=rng.uniform(-3,3,(1,2))
        if not inside(x):
            x = _add_noise(x, noise_std, rng)
            pos.append(x)
    for _ in range(N_neg):
        # 在圆内均匀
        rho=r*np.sqrt(rng.rand()); phi=2*np.pi*rng.rand()
        u=np.array([[rho*np.cos(phi),rho*np.sin(phi)]])
        x=u+np.array([cx,cy])
        x=_add_noise(x, noise_std, rng)
        neg.append(x)
    return np.vstack(pos).astype(np.float32), np.vstack(neg).astype(np.float32)

def sample_ellipse(center=(0.0,0.0), a=1.4, b=0.8, theta_deg=30.0, N_pos=1000, N_neg=1000, seed=0, noise_std=0.05):
    rng=np.random.RandomState(seed); cx,cy=center
    th=np.deg2rad(theta_deg); c,s=np.cos(th),np.sin(th)
    R=np.array([[c,-s],[s,c]],dtype=np.float32)
    R_T=np.array([[c, s],[-s, c]],dtype=np.float32)
    def inside(x):
        y=x-np.array([cx,cy])
        yloc=y@R_T.T
        return (yloc[:,0]/a)**2+(yloc[:,1]/b)**2<=1.0
    pos,neg=[],[]
    while len(pos)<N_pos:
        x=rng.uniform(-3,3,(1,2))
        x=_add_noise(x, noise_std, rng)
        if not inside(x): pos.append(x)
    for _ in range(N_neg):
        # 先在局部矩形采，拒绝直到落在椭圆内
        while True:
            u=rng.uniform([-a,-b],[a,b],(1,2))
            if (u[0,0]/a)**2+(u[0,1]/b)**2 <= 1.0:
                break
        x=u@R.T + np.array([cx,cy])
        x=_add_noise(x, noise_std, rng)
        neg.append(x)
    return np.vstack(pos).astype(np.float32), np.vstack(neg).astype(np.float32)

def sample_rectangle(center=(0.0,0.0), w=1.2, h=0.7, theta_deg=20.0, N_pos=1000, N_neg=1000, seed=0, noise_std=0.1):
    rng=np.random.RandomState(seed); cx,cy=center
    th=np.deg2rad(theta_deg); c,s=np.cos(th),np.sin(th)
    R=np.array([[c,-s],[s,c]],dtype=np.float32)
    R_T=np.array([[c, s],[-s, c]],dtype=np.float32)
    def inside(x):
        y=x-np.array([cx,cy]); yloc=y@R_T.T
        return (np.abs(yloc[:,0])<=w)&(np.abs(yloc[:,1])<=h)
    pos,neg=[],[]
    while len(pos)<N_pos:
        x=rng.uniform(-3,3,(1,2))
        if not inside(x):
            x = _add_noise(x, noise_std, rng)
            pos.append(x)
    for _ in range(N_neg):
        u=rng.uniform([-w,-h],[w,h],(1,2))
        x=u@R.T + np.array([cx,cy])
        x=_add_noise(x, noise_std, rng)
        neg.append(x)
    return np.vstack(pos).astype(np.float32), np.vstack(neg).astype(np.float32)

def sample_rect_circle_union(
    center_rect=(-0.6,0.0), w=1.0, h=0.5, theta_deg=0.0,
    center_circle=(2.3,0.3), r=0.5,
    N_pos=1000, N_neg=1000, seed=0, noise_std=0.1
):
    rng = np.random.RandomState(seed)
    cx,cy = center_rect; th=np.deg2rad(theta_deg)
    c,s=np.cos(th),np.sin(th)
    R  = np.array([[c,-s],[s,c]],dtype=np.float32)  # 局部→全局
    R_T= np.array([[c, s],[-s, c]],dtype=np.float32)  # 全局→局部
    ccx, ccy = center_circle

    def inside_rect(x):
        y = x - np.array([cx,cy])
        yloc = y @ R_T.T
        return (np.abs(yloc[:,0])<=w) & (np.abs(yloc[:,1])<=h)

    def inside_circle(x):
        return np.linalg.norm(x - np.array([ccx,ccy]), axis=1) <= r

    def inside_union(x):
        return inside_rect(x) | inside_circle(x)

    pos, neg = [], []
    while len(pos) < N_pos:
        x = rng.uniform(-3,3,(1,2))
        if not inside_union(x):
            x = _add_noise(x, noise_std, rng)
            pos.append(x)

    for _ in range(N_neg):
        if rng.rand() < 0.5:
            # 矩形内部均匀
            u = rng.uniform([-w,-h],[w,h],(1,2))
            x = u @ R.T + np.array([cx,cy])
        else:
            # 圆内部均匀
            rho = r*np.sqrt(rng.rand()); phi = 2*np.pi*rng.rand()
            u = np.array([[rho*np.cos(phi), rho*np.sin(phi)]])
            x = u + np.array([ccx,ccy])
        x = _add_noise(x, noise_std, rng)
        neg.append(x)

    return np.vstack(pos).astype(np.float32), np.vstack(neg).astype(np.float32)

# ===================================================
# 基础网络
# ===================================================
class MLP(nn.Module):
    def __init__(self,in_dim=1,hidden=(64,64),out_dim=3,activation=nn.ReLU):
        super().__init__()
        layers=[]; last=in_dim
        for h in hidden:
            layers+=[nn.Linear(last,h),activation()]; last=h
        layers+=[nn.Linear(last,out_dim)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class CircleConstraintNet(nn.Module):
    def __init__(self,hidden=(64,64),init=(0.0,0.0,1.0)):
        super().__init__()
        self.head=MLP(1,hidden,3); self.softplus=nn.Softplus()
        self.register_buffer("const_in",torch.ones(1,1))
        with torch.no_grad():
            for m in self.head.modules():
                if isinstance(m,nn.Linear): nn.init.zeros_(m.weight)
            cx0,cy0,r0=init; rho0=np.log(np.exp(r0**2)-1.0)
            last=[m for m in self.head.modules() if isinstance(m,nn.Linear)][-1]
            last.bias.copy_(torch.tensor([cx0,cy0,rho0]))
    def params_numpy(self):
        with torch.no_grad():
            out=self.head(self.const_in).squeeze(0); cx,cy,rho=out
            r=np.sqrt(self.softplus(rho).item()); return (cx.item(),cy.item(),r)
    def forward(self,x):
        theta=self.head(self.const_in).squeeze(0); cx,cy,rho=theta
        s=self.softplus(rho); diff=x-torch.stack([cx,cy])[None,:]
        sq=torch.sum(diff*diff,dim=1,keepdim=True); return sq-s

# ===================================================
# 组合器
# ===================================================
class CompositeConstraintNet(nn.Module):
    def __init__(self,components,mode="union"):
        super().__init__()
        self.components=nn.ModuleList(components)
        assert mode in("union","intersection")
        self.mode=mode
    def forward(self,x):
        vals=[c(x) for c in self.components]
        f_all=torch.stack(vals,dim=-1) # (N,1,M)
        if self.mode=="union":
            return torch.min(f_all,dim=-1).values
        else:
            return torch.max(f_all,dim=-1).values

# ===================================================
# 损失
# ===================================================
def logistic_margin_loss(f,y,tau=0.07):
    return torch.log1p(torch.exp(-y*f/tau)).mean()

# ===================================================
# 示例运行
# ===================================================
if __name__=="__main__":
    SHAPE="union"  # circle | ellipse | rectangle | union
    if SHAPE=="circle":
        true_center=(0.3,-0.1); true_r=1.25
        X_pos,X_neg=sample_circle(center=true_center,r=true_r,N_pos=2000,N_neg=2000,seed=42)
        net=CircleConstraintNet(init=(0.0,0.0,0.6))
        gt_params=(true_center[0],true_center[1],true_r)
    elif SHAPE=="union":
        X_pos,X_neg=sample_rect_circle_union()
        net=CompositeConstraintNet([
            RectangleConstraintNet(init=(0.0,0.0,1.0,0.5,0.0)),
            CircleConstraintNet(init=(0.3,0.0,0.8))
        ],mode="union")
        gt_params=None

    X=np.vstack([X_pos,X_neg]).astype(np.float32)
    y=np.vstack([np.ones((len(X_pos),1),dtype=np.float32),
                 -np.ones((len(X_neg),1),dtype=np.float32)])
    idx=np.random.permutation(len(X)); X=torch.tensor(X[idx]); y=torch.tensor(y[idx])

    opt=optim.Adam(net.parameters(),lr=2e-3); tau=0.05
    for it in range(1,501):
        perm=torch.randperm(len(X)); Xb,yb=X[perm[:2048]],y[perm[:2048]]
        opt.zero_grad(); f=net(Xb); loss=logistic_margin_loss(f,yb,tau)
        loss.backward(); opt.step()
        if it%100==0: print(f"[{SHAPE}] it={it} | loss={loss.item():.4f}")

    if SHAPE=="circle":
        est_params=net.params_numpy()
    elif SHAPE=="union":
        est_params=[["rectangle"]+list(net.components[0].params_numpy()),
                    ["circle"]+list(net.components[1].params_numpy())]

    plot_dataset_and_shape(X_pos,X_neg,shape=SHAPE,gt_params=gt_params,est_params=est_params,
                           title=f"{SHAPE.capitalize()} — learned vs GT")

