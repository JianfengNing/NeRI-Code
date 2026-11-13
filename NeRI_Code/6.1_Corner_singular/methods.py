





from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["neri", "pinn", "wan", "neri-gd", "neri-two-layer-gd", "fd"],
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
import math
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class Parameters:
    

    
    method: str = 'neri'
    seed: int = 0  
    N: int = 10000  
    Nb: int = 256  
    epochs: int | None = None  
    width: int | None = None
    blocks: int = 2  
    act: str = 'tanh'  
    learning_rate: float | None = None
    optimizer: str | None = None
    
    scheduler: str | None = None 
    step_size: int | None = None  
    test_step_size: int = 1000  
    gamma: float = 0.8  
    cosine_t_max: int | None = None  
    eta_min: float = 1e-8  
    weight_decay: float | None = None  
    momentum: float = 0.0  
    boundary_weight: float = 50.0
    resample: bool = True  
    eval_every: int = 1  
    grid_n: int = 201  
    show_plots: bool = True  
    fd_h: float = 0.005  
    

    def __post_init__(self):
        if self.method not in (
            "neri",
            "pinn",
            "wan",
            "neri-gd",
            "neri-two-layer-gd",
            "fd",
        ):
            raise ValueError(f"Unknown method: {self.method}")
        is_gd = self.method in ("neri-gd", "neri-two-layer-gd")
        defaults = (15000, 100, 0.001)
        if self.method == "neri-gd":
            defaults = (100000, 200, 0.005)
        elif self.method == "neri-two-layer-gd":
            defaults = (100000, 100, 0.005)
        if self.epochs is None:
            self.epochs = defaults[0]
        if self.width is None:
            self.width = defaults[1]
        if self.learning_rate is None:
            self.learning_rate = defaults[2]
        if self.optimizer is None:
            self.optimizer = "sgd" if is_gd else "adam"
        if self.scheduler is None:
            self.scheduler = "constant" if self.method == "neri-two-layer-gd" else "cosine"
        if self.step_size is None:
            self.step_size = 500
        self.optimizer = self.optimizer.lower()
        self.scheduler = self.scheduler.lower()
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.scheduler not in ("step", "cosine", "constant"):
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
        if self.weight_decay is None:
            self.weight_decay = 0.01 if self.optimizer == "adamw" else 0.0


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OMEGA_MEAS = 4.0
BDRY_MEAS = 9.0
WAN_EPS = 1e-10
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def data_path(filename):
    
    path = Path(filename)
    if path.is_absolute():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / path


def make_optimizer(model_parameters, parameters):
    
    options = dict(lr=parameters.learning_rate, weight_decay=parameters.weight_decay)
    if parameters.optimizer == "adam":
        return torch.optim.Adam(model_parameters, **options)
    if parameters.optimizer == "adamw":
        return torch.optim.AdamW(model_parameters, **options)
    if parameters.optimizer == "sgd":
        return torch.optim.SGD(
            model_parameters, momentum=parameters.momentum, nesterov=False, **options
        )
    raise ValueError(f"Unknown optimizer: {parameters.optimizer}")


def make_scheduler(optimizer, parameters, *, total_steps, step_size=None):
    
    if parameters.scheduler == "step":
        interval = parameters.step_size if step_size is None else step_size
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=interval, gamma=parameters.gamma
        )
    if parameters.scheduler == "cosine":
        period = (
            total_steps if parameters.cosine_t_max is None else parameters.cosine_t_max
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=period, eta_min=parameters.eta_min
        )
    if parameters.scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    raise ValueError(f"Unknown scheduler: {parameters.scheduler}")


def step_scheduler(scheduler, epoch=None):
    
    if epoch is not None and isinstance(
        scheduler,
        (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.MultiStepLR),
    ):
        scheduler.step(epoch)
    else:
        scheduler.step()


PINN_BDRY_MEAS = 8.0


def method_label(method):
    
    return {
        "neri": "NeRI",
        "pinn": "PINN",
        "wan": "WAN",
        "neri-gd": "NeRI-GD",
        "neri-two-layer-gd": "NeRI-two-layer-GD",
        "fem": "FEM",
        "fd": "FD",
    }[method]


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def u_exact(x: torch.Tensor):
    xx = x[:, 0]
    yy = x[:, 1]
    r = (xx**2 + yy**2) ** 0.5
    u = torch.sqrt(torch.abs(r - xx) / 2)
    return u


def f_rhs(x: torch.Tensor):
    xx = x[:, 0]
    yy = x[:, 1]
    r = (xx**2 + yy**2) ** 0.5
    f = -torch.sqrt(torch.abs(r - xx) / 2) / (2 * r)
    return f


def laplace_u(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(x.shape[1]):
        grad_i = grad[:, i]
        second = torch.autograd.grad(
            grad_i, x, torch.ones_like(grad_i), create_graph=True
        )[0][:, i]
        lap = lap + second
    return lap


def sample_interior(N: int, eps=1e-06) -> torch.Tensor:
    x = 2.0 * torch.rand(N, 2, device=device) - 1.0
    r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    mask = r > eps
    return x[mask]


def sample_boundary_per_face(N: int, d: int, device=device):
    x_list = []
    for j in range(d):
        pts0 = 2.0 * torch.rand(N, d, device=device) - 1.0
        pts0[:, j] = -1.0
        x_list.append(pts0)
        pts1 = 2.0 * torch.rand(N, d, device=device) - 1.0
        pts1[:, j] = 1.0
        x_list.append(pts1)
    ptsi = torch.rand(N, d, device=device)
    ptsi[:, 1] = 0
    x_list.append(ptsi)
    x = torch.cat(x_list, dim=0)
    return x


def grad_u(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]


class ResBlock(nn.Module):

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.act(self.fc1(z))
        out = self.act(self.fc2(out))
        return out + z


class FDRR(nn.Module):

    def __init__(self, in_dim: int = 2, width: int = 80, blocks: int = 4):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, width)
        self.act = nn.Tanh()
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(blocks)])
        self.fcout = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.fc0(x))
        for b in self.blocks:
            z = b(z)
        return self.fcout(z).squeeze(-1)


def set_seed_two_layer(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def u_exact_two_layer(x: torch.Tensor) -> torch.Tensor:
    xx, yy = (x[:, 0], x[:, 1])
    r = torch.sqrt(xx**2 + yy**2)
    return torch.sqrt(torch.abs(r - xx) / 2.0)


def f_rhs_two_layer(x: torch.Tensor) -> torch.Tensor:
    xx, yy = (x[:, 0], x[:, 1])
    r = torch.sqrt(xx**2 + yy**2)
    return -torch.sqrt(torch.abs(r - xx) / 2.0) / (2.0 * r)


def sample_interior_two_layer(N: int, eps: float = 1e-06) -> torch.Tensor:
    x = 2.0 * torch.rand(N, 2, device=device) - 1.0
    r = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    return x[r > eps]


def sample_boundary_per_face_two_layer(N: int, d: int, device=device):
    if d != 2:
        raise ValueError("Boundary sampling for the slit domain requires d=2.")
    x_list = []
    for j in range(d):
        pts0 = 2.0 * torch.rand(N, d, device=device) - 1.0
        pts0[:, j] = -1.0
        x_list.append(pts0)
        pts1 = 2.0 * torch.rand(N, d, device=device) - 1.0
        pts1[:, j] = 1.0
        x_list.append(pts1)
    ptsi = torch.rand(N, d, device=device)
    ptsi[:, 1] = 0.0
    x_list.append(ptsi)
    return torch.cat(x_list, dim=0)


class TwoLayerFNN(nn.Module):
    

    def __init__(self, in_dim: int = 2, width: int = 100):
        super().__init__()
        if width < 1:
            raise ValueError("width must be a positive integer.")
        self.fc1 = nn.Linear(in_dim, width, bias=True)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(width, 1, bias=False)
        self.scale = 1.0 / math.sqrt(width)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.fc1(x))
        return self.scale * self.fc2(z).squeeze(-1)


def loss_neri(
    u: torch.Tensor, x: torch.Tensor, ube, ub, boundary_weight=50.0
) -> torch.Tensor:
    
    ux = grad_u(u, x)
    vx = ux.detach()
    f = f_rhs(x).detach()
    integrand = (vx * ux).sum(dim=1) + vx[:, 0] * u - f * u
    return OMEGA_MEAS * torch.mean(
        integrand
    ) + boundary_weight * BDRY_MEAS * torch.mean((ub - ube) ** 2)


def loss_pinn(
    u: torch.Tensor, x: torch.Tensor, ube, ub, boundary_weight=50.0
) -> torch.Tensor:
    
    ux = grad_u(u, x)
    uxx = laplace_u(u, x)
    f = f_rhs(x).detach()
    rr = (-uxx + ux[:, 0] - f) ** 2
    return OMEGA_MEAS * torch.mean(rr) + boundary_weight * PINN_BDRY_MEAS * torch.mean(
        (ub - ube) ** 2
    )


def loss_wan_solution(u, v, x, ube, ub, boundary_weight=50.0):
    
    ux = grad_u(u, x)
    vx = grad_u(v, x)
    v = v.detach()
    vx = vx.detach()
    f = f_rhs(x).detach()
    integrand = (ux * vx).sum(dim=1) + ux[:, 0] * v - f * v
    return torch.log(
        OMEGA_MEAS * torch.mean(integrand) ** 2 + WAN_EPS
    ) + boundary_weight * BDRY_MEAS * torch.mean((ub - ube) ** 2)


def loss_wan_test(u, v, x, vb, boundary_weight=50.0):
    
    ux = grad_u(u, x)
    vx = grad_u(v, x)
    ux = ux.detach()
    f = f_rhs(x).detach()
    integrand = (ux * vx).sum(dim=1) + ux[:, 0] * v - f * v
    loss1 = -torch.log(OMEGA_MEAS * torch.mean(integrand) ** 2 + WAN_EPS) + torch.log(
        OMEGA_MEAS * torch.mean((vx * vx).sum(dim=1)) + WAN_EPS
    )
    loss2 = boundary_weight * BDRY_MEAS * torch.mean(vb**2)
    return loss1 + loss2


def loss_neri_two_layer_gd(
    u: torch.Tensor, x: torch.Tensor, ube, ub, boundary_weight
) -> torch.Tensor:
    
    ux = grad_u(u, x)
    vx = ux.detach()
    f = f_rhs_two_layer(x).detach()
    integrand = (vx * ux).sum(dim=1) + vx[:, 0] * u - f * u
    loss_interior = OMEGA_MEAS * torch.mean(integrand)
    loss_boundary = boundary_weight * BDRY_MEAS * torch.mean((ub - ube) ** 2)
    return loss_interior + loss_boundary


def plot_comparison(Xn, Yn, u_nn_np, u_ex_np, diff_np):
    
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131)
    contour = ax1.contourf(Xn, Yn, u_nn_np, cmap="jet", levels=200)
    ax1.set_title("U_nn", fontsize=14, pad=25)
    ax1.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax1)
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(Xn, Yn, u_ex_np, cmap="jet", levels=200)
    ax2.set_title("U_exact", fontsize=14, pad=25)
    ax2.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax2)
    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(Xn, Yn, diff_np, cmap="jet", levels=200)
    ax3.set_title("U_exact", fontsize=14, pad=25)
    ax3.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax3)
    plt.tight_layout()
    plt.savefig(data_path(f"OANI.png"), dpi=200, bbox_inches="tight")
    plt.show()


def train_neri(seed: int | None = None, params=None):
    
    P = Parameters(method="neri") if params is None else params
    seed = P.seed if seed is None else seed
    set_seed(seed)
    z_u = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)

    def u_model(x):
        return z_u(x)

    grid_n = 201
    xs = torch.linspace(-1, 1, grid_n, device=device)
    ys = torch.linspace(-1, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    ute = None
    U = u_exact(pts)
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        x = sample_interior(P.N)
        x.requires_grad_(True)
        xb = sample_boundary_per_face(P.Nb, 2)
        u = u_model(x)
        ub = u_model(xb)
        ube = u_exact(xb)
        
        sample_interior(1)
        Loss = loss_neri(u, x, ube, ub, boundary_weight=P.boundary_weight)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u)
        with torch.no_grad():
            ute = u_model(pts)
            error_mean_square = torch.mean((ute - U) ** 2)
            l2error = torch.sqrt(error_mean_square / reference_mean_square)
            error_history.append(l2error.item())
            if ep % 1000 == 0 or ep == P.epochs - 1:
                l2 = (torch.sqrt(error_mean_square) / reference_norm).item()
                linf = torch.max(torch.abs(ute - U)).item()
                dt = time.time() - t0
                print(
                    f"[{method_label(P.method)}] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s"
                )
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        u_nn = ute.reshape(grid_n, grid_n).cpu()
        u_ex = U.reshape(grid_n, grid_n).cpu()
        diff = (u_nn - u_ex).abs()
        print(
            "[grid] mean|diff|=", diff.mean().item(), " max|diff|=", diff.max().item()
        )
        Xn = X.cpu().numpy()
        Yn = Y.cpu().numpy()
        u_nn_np = u_nn.numpy()
        u_ex_np = u_ex.numpy()
        diff_np = diff.numpy()
        np.savez(
            data_path("NeRI.npz"),
            Xn=Xn,
            Yn=Yn,
            u_nn=u_nn_np,
            uex=u_ex_np,
            error_history=error_history,
        )
        plot_comparison(Xn, Yn, u_nn_np, u_ex_np, diff_np)
    return z_u


def train_pinn(seed: int | None = None, params=None):
    
    P = Parameters(method="pinn") if params is None else params
    seed = P.seed if seed is None else seed
    set_seed(seed)
    z_u = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)

    def u_model(x):
        return z_u(x)

    grid_n = 201
    xs = torch.linspace(-1, 1, grid_n, device=device)
    ys = torch.linspace(-1, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    ute = None
    U = u_exact(pts)
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        x = sample_interior(P.N)
        x.requires_grad_(True)
        xb = sample_boundary_per_face(P.Nb, 2)
        u = u_model(x)
        ub = u_model(xb)
        ube = u_exact(xb)
        
        sample_interior(1)
        Loss = loss_pinn(u, x, ube, ub, boundary_weight=P.boundary_weight)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u)
        with torch.no_grad():
            ute = u_model(pts)
            error_mean_square = torch.mean((ute - U) ** 2)
            l2error = torch.sqrt(error_mean_square / reference_mean_square)
            error_history.append(l2error.item())
            if ep % 1000 == 0 or ep == P.epochs - 1:
                l2 = (torch.sqrt(error_mean_square) / reference_norm).item()
                linf = torch.max(torch.abs(ute - U)).item()
                dt = time.time() - t0
                print(
                    f"[{method_label(P.method)}] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s"
                )
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        u_nn = ute.reshape(grid_n, grid_n).cpu()
        u_ex = U.reshape(grid_n, grid_n).cpu()
        diff = (u_nn - u_ex).abs()
        print(
            "[grid] mean|diff|=", diff.mean().item(), " max|diff|=", diff.max().item()
        )
        Xn = X.cpu().numpy()
        Yn = Y.cpu().numpy()
        u_nn_np = u_nn.numpy()
        u_ex_np = u_ex.numpy()
        diff_np = diff.numpy()
        np.savez(
            data_path("PINN.npz"),
            Xn=Xn,
            Yn=Yn,
            u_nn=u_nn_np,
            uex=u_ex_np,
            error_history=error_history,
        )
        plot_comparison(Xn, Yn, u_nn_np, u_ex_np, diff_np)
    return z_u


def train_wan(seed: int | None = None, params=None):
    
    P = Parameters(method="wan") if params is None else params
    seed = P.seed if seed is None else seed
    set_seed(seed)
    z_u = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)
    z_v = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_v = make_optimizer(z_v.parameters(), P)
    sch_v = make_scheduler(
        opt_v, P, total_steps=2 * P.epochs, step_size=P.test_step_size
    )

    def u_model(x):
        return z_u(x)

    def v_model(x):
        return z_v(x)

    grid_n = 201
    xs = torch.linspace(-1, 1, grid_n, device=device)
    ys = torch.linspace(-1, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    ute = None
    U = u_exact(pts)
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        for _ in range(2):
            x = sample_interior(P.N).requires_grad_(True)
            u = u_model(x)
            v = v_model(x)
            xb = sample_boundary_per_face(P.Nb, 2)
            vb = v_model(xb)
            loss_v = loss_wan_test(u, v, x, vb, boundary_weight=P.boundary_weight)
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(z_v.parameters(), 10.0)
            opt_v.step()
            step_scheduler(sch_v)
        for _ in range(1):
            x = sample_interior(P.N).requires_grad_(True)
            u = u_model(x)
            v = v_model(x)
            xb = sample_boundary_per_face(P.Nb, 2)
            ub = u_model(xb)
            ube = u_exact(xb)
            loss_u = loss_wan_solution(
                u, v, x, ube, ub, boundary_weight=P.boundary_weight
            )
            opt_u.zero_grad(set_to_none=True)
            loss_u.backward()
            torch.nn.utils.clip_grad_norm_(z_u.parameters(), 10.0)
            opt_u.step()
            step_scheduler(sch_u)
        with torch.no_grad():
            ute = u_model(pts)
            error_mean_square = torch.mean((ute - U) ** 2)
            l2error = torch.sqrt(error_mean_square / reference_mean_square)
            error_history.append(l2error.item())
            if ep % 1000 == 0 or ep == P.epochs - 1:
                l2 = (torch.sqrt(error_mean_square) / reference_norm).item()
                linf = torch.max(torch.abs(ute - U)).item()
                dt = time.time() - t0
                print(
                    f"[{method_label(P.method)}] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s"
                )
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        u_nn = ute.reshape(grid_n, grid_n).cpu()
        u_ex = U.reshape(grid_n, grid_n).cpu()
        diff = (u_nn - u_ex).abs()
        print(
            "[grid] mean|diff|=", diff.mean().item(), " max|diff|=", diff.max().item()
        )
        Xn = X.cpu().numpy()
        Yn = Y.cpu().numpy()
        u_nn_np = u_nn.numpy()
        u_ex_np = u_ex.numpy()
        diff_np = diff.numpy()
        np.savez(
            data_path("WAN.npz"),
            Xn=Xn,
            Yn=Yn,
            u_nn=u_nn_np,
            uex=u_ex_np,
            error_history=error_history,
        )
        plot_comparison(Xn, Yn, u_nn_np, u_ex_np, diff_np)
    return (z_u, z_v)


def train_neri_gd(seed: int | None = None, params=None):
    
    P = Parameters(method="neri-gd") if params is None else params
    seed = P.seed if seed is None else seed
    set_seed(seed)
    z_u = TwoLayerFNN(in_dim=2, width=P.width).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)

    def u_model(x):
        return z_u(x)

    grid_n = 201
    xs = torch.linspace(-1, 1, grid_n, device=device)
    ys = torch.linspace(-1, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    ute = None
    U = u_exact(pts)
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        x = sample_interior(P.N)
        x.requires_grad_(True)
        xb = sample_boundary_per_face(P.Nb, 2)
        u = u_model(x)
        ub = u_model(xb)
        ube = u_exact(xb)
        
        sample_interior(1)
        Loss = loss_neri(u, x, ube, ub, boundary_weight=P.boundary_weight)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u)
        with torch.no_grad():
            ute = u_model(pts)
            error_mean_square = torch.mean((ute - U) ** 2)
            l2error = torch.sqrt(error_mean_square / reference_mean_square)
            error_history.append(l2error.item())
            if ep % 1000 == 0 or ep == P.epochs - 1:
                l2 = (torch.sqrt(error_mean_square) / reference_norm).item()
                linf = torch.max(torch.abs(ute - U)).item()
                dt = time.time() - t0
                print(
                    f"[{method_label(P.method)}] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s"
                )
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        u_nn = ute.reshape(grid_n, grid_n).cpu()
        u_ex = U.reshape(grid_n, grid_n).cpu()
        diff = (u_nn - u_ex).abs()
        print(
            "[grid] mean|diff|=", diff.mean().item(), " max|diff|=", diff.max().item()
        )
        Xn = X.cpu().numpy()
        Yn = Y.cpu().numpy()
        u_nn_np = u_nn.numpy()
        u_ex_np = u_ex.numpy()
        diff_np = diff.numpy()
        np.savez(
            data_path(f"NeRI_GD_{P.width}.npz"),
            Xn=Xn,
            Yn=Yn,
            u_nn=u_nn_np,
            uex=u_ex_np,
            error_history=error_history,
        )
        plot_comparison(Xn, Yn, u_nn_np, u_ex_np, diff_np)
    return z_u


def train_neri_two_layer_gd(seed: int | None = None, params=None):
    
    P = Parameters(method="neri-two-layer-gd") if params is None else params
    seed = P.seed if seed is None else seed
    set_seed_two_layer(seed)
    if min(P.N, P.Nb, P.epochs, P.eval_every) < 1 or P.grid_n < 2:
        raise ValueError(
            "Sample counts, epochs, and evaluation interval must be positive; grid_n must be at least 2."
        )
    if not math.isfinite(P.learning_rate) or P.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(P.boundary_weight) or P.boundary_weight < 0:
        raise ValueError("boundary_weight must be finite and nonnegative.")
    z_u = TwoLayerFNN(in_dim=2, width=P.width).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)

    def u_model(x):
        return z_u(x)

    grid_n = P.grid_n
    xs = torch.linspace(-1, 1, grid_n, device=device)
    ys = torch.linspace(-1, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        U = u_exact_two_layer(pts)
        U_norm = torch.sqrt(torch.mean(U**2))
    error_history, error_epochs, flow_time_history = ([], [], [])
    print("Number of parameters:", sum((q.numel() for q in z_u.parameters())))
    t0 = time.time()
    flow_time = 0.0
    for ep in range(P.epochs):
        if P.resample or ep == 0:
            x_data = sample_interior_two_layer(P.N)
            if x_data.shape[0] == 0:
                raise RuntimeError(
                    "Interior sample is empty; increase N or reduce eps."
                )
            xb = sample_boundary_per_face_two_layer(P.Nb, 2)
            ube = u_exact_two_layer(xb)
        x = x_data.detach().requires_grad_(True)
        opt_u.zero_grad(set_to_none=True)
        u = u_model(x)
        ub = u_model(xb)
        Loss = loss_neri_two_layer_gd(u, x, ube, ub, P.boundary_weight)
        Loss.backward()
        if P.scheduler != "constant":
            flow_time += opt_u.param_groups[0]["lr"]
        opt_u.step()
        step_scheduler(sch_u)
        flow_t = (ep + 1) * P.learning_rate if P.scheduler == "constant" else flow_time
        log_now = ep % 1000 == 0 or ep == P.epochs - 1
        eval_now = ep % P.eval_every == 0 or log_now
        if eval_now:
            with torch.no_grad():
                ute = u_model(pts)
                l2 = (torch.sqrt(torch.mean((ute - U) ** 2)) / U_norm).item()
                linf = torch.max(torch.abs(ute - U)).item()
            error_history.append(l2)
            error_epochs.append(ep + 1)
            flow_time_history.append(flow_t)
        if log_now:
            print(
                f"[{method_label(P.method)}] ep={ep + 1:5d}  L2={l2:.3e}  Linf={linf:.3e}  Jk={Loss.item():.3e}  flow_t={flow_t:.3f}  wall_t={time.time() - t0:.1f}s"
            )
    with torch.no_grad():
        u_nn = ute.reshape(grid_n, grid_n).cpu().numpy()
        u_ex = U.reshape(grid_n, grid_n).cpu().numpy()
        Xn, Yn = (X.cpu().numpy(), Y.cpu().numpy())
    diff = np.abs(u_nn - u_ex)
    print("[grid] mean|diff|=", diff.mean(), "max|diff|=", diff.max())
    out = DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"NeRI_GD_{P.width}.npz",
        Xn=Xn,
        Yn=Yn,
        u_nn=u_nn,
        uex=u_ex,
        error_history=np.asarray(error_history),
        error_epochs=np.asarray(error_epochs),
        flow_time_history=np.asarray(flow_time_history),
        width=P.width,
        lr_s=P.learning_rate,
        resample=P.resample,
        boundary_weight=P.boundary_weight,
    )
    torch.save(z_u.state_dict(), out / "NeRI_two_layer_GD.pt")
    if P.show_plots:
        import matplotlib.pyplot as plt

        for values, title, filename in [
            (u_nn, "NeRI: two-layer FNN + GD", "NeRI_two_layer_GD.png"),
            (u_ex, "Exact solution", "NeRI_exact.png"),
            (diff, "Absolute error", "NeRI_two_layer_GD_error.png"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contourf(Xn, Yn, values, cmap="jet", levels=200)
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(contour, ax=ax)
            fig.tight_layout()
            fig.savefig(out / filename, dpi=200, bbox_inches="tight")
        plt.show()
    return z_u


def run_fd(parameters=None):
    
    import matplotlib.pyplot as plt
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    P = Parameters(method="fd") if parameters is None else parameters
    h = P.fd_h
    x = np.arange(-1, 1 + h, h)
    y = np.arange(-1, 1 + h, h)
    nx = len(x)
    ny = len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    inside_square = (X > -1) & (X < 1) & (Y > -1) & (Y < 1)
    slit = (X > 0) & (X < 1) & (np.abs(Y) < 1e-12)
    active = inside_square & ~slit
    idx = -np.ones((nx, ny), dtype=int)
    idx[active] = np.arange(np.sum(active))
    N = np.sum(active)
    print("Number of unknowns:", N)
    A = lil_matrix((N, N))
    b = np.ones(N)
    for i in range(nx):
        for j in range(ny):
            if not active[i, j]:
                continue
            row = idx[i, j]
            A[row, row] = 4.0 / h**2
            ni, nj = (i + 1, j)
            coeff = -1.0 / h**2 + 1.0 / (2.0 * h)
            if active[ni, nj]:
                A[row, idx[ni, nj]] += coeff
            ni, nj = (i - 1, j)
            coeff = -1.0 / h**2 - 1.0 / (2.0 * h)
            if active[ni, nj]:
                A[row, idx[ni, nj]] += coeff
            ni, nj = (i, j + 1)
            coeff = -1.0 / h**2
            if active[ni, nj]:
                A[row, idx[ni, nj]] += coeff
            ni, nj = (i, j - 1)
            coeff = -1.0 / h**2
            if active[ni, nj]:
                A[row, idx[ni, nj]] += coeff
    A = A.tocsr()
    u_vec = spsolve(A, b)
    U = np.full((nx, ny), np.nan)
    U[active] = u_vec
    U[~active] = 0.0
    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, U, cmap="jet", levels=300)
    plt.colorbar(label="u(x,y)")
    plt.plot([0, 1], [0, 0], "k-", linewidth=3, label="slit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution of $-\\Delta u + u_x = 1$")
    plt.axis("equal")
    plt.legend()
    plt.show()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, U, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title("Finite Difference Solution of $-\\Delta u + u_x = 1$")
    plt.show()


def main(parameters: Parameters | None = None):
    parameters = Parameters() if parameters is None else parameters
    if parameters.method not in (
        "neri",
        "pinn",
        "wan",
        "neri-gd",
        "neri-two-layer-gd",
        "fd",
    ):
        raise ValueError(f"Unknown method: {parameters.method}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if parameters.method == "fd":
        return run_fd(parameters)
    print("Device:", device)
    print("Running...")
    methods = {
        "neri": train_neri,
        "pinn": train_pinn,
        "wan": train_wan,
        "neri-gd": train_neri_gd,
        "neri-two-layer-gd": train_neri_two_layer_gd,
    }
    return methods[parameters.method](seed=parameters.seed, params=parameters)


if __name__ == "__main__":
    overrides = {
        name: value for name, value in vars(_args).items() if value is not None
    }
    main(Parameters(**overrides))

