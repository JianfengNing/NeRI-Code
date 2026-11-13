

import argparse
from pathlib import Path
import random
import time
from dataclasses import dataclass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("neri", "pinn", "wan"),
        default=None,
        help="Override Parameters.method.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override Parameters.seed."
    )
    return parser.parse_args()



if __name__ == "__main__":
    _args = parse_args()

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Parameters:
    

    
    method: str = 'pinn'  
    seed: int = 0  
    N: int = 10000  
    Nb: int = 256  
    epochs: int = 20000  
    width: int = 100  
    blocks: int = 2  
    act: str = 'tanh'  
    boundary_weight: float = 100  
    tau: float = 0.01  
    wan_regularization: float = 0.0005  
    learning_rate: float = 0.001  
    optimizer: str | None = None  
    
    scheduler: str = 'cosine'
    step_size: int = 500  
    test_step_size: int = 1000  
    gamma: float = 0.8  
    cosine_t_max: int | None = None  
    eta_min: float = 1e-8  
    weight_decay: float | None = None  
    momentum: float = 0.0  
    


DATA_DIR = Path(__file__).resolve().parent / "data"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

k = 8
coeff_ns = 1.0
p = 2


d = 2
eps = 1e-10

OMEGA_MEAS = 4.0
BDRY_MEAS = 8.0


def make_optimizer(model_parameters, parameters):
    
    name = parameters.optimizer
    if name is None:
        name = "adamw" if parameters.method == "pinn" else "adam"
    name = name.lower()
    weight_decay = parameters.weight_decay
    if weight_decay is None:
        weight_decay = 0.01 if name == "adamw" else 0.0
    options = {"lr": parameters.learning_rate, "weight_decay": weight_decay}
    if name == "adam":
        return torch.optim.Adam(model_parameters, **options)
    if name == "adamw":
        return torch.optim.AdamW(model_parameters, **options)
    if name == "sgd":
        return torch.optim.SGD(
            model_parameters, momentum=parameters.momentum, **options
        )
    raise ValueError(f"Unknown optimizer: {name}. Choose adam, adamw, or sgd.")


def make_scheduler(optimizer, parameters, *, total_steps, step_size=None):
    
    name = parameters.scheduler.lower()
    if name == "step":
        interval = parameters.step_size if step_size is None else step_size
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=interval, gamma=parameters.gamma
        )
    if name == "cosine":
        horizon = parameters.cosine_t_max
        if horizon is None:
            horizon = total_steps
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=parameters.eta_min
        )
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    raise ValueError(f"Unknown scheduler: {name}. Choose step, cosine, or constant.")


def step_scheduler(scheduler, epoch=None):
    
    epoch_schedulers = (
        torch.optim.lr_scheduler.StepLR,
        torch.optim.lr_scheduler.MultiStepLR,
    )
    if epoch is not None and isinstance(scheduler, epoch_schedulers):
        scheduler.step(epoch)
    else:
        scheduler.step()


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def u_exact(x):
    r2 = torch.sum(x**2, dim=1)
    s = torch.clamp(1.0 - k * r2, min=0.0)
    return s**p + 1.0


def f_rhs(x):
    dim = x.shape[1]
    r2 = torch.sum(x**2, dim=1)
    f = torch.zeros_like(r2)
    inside = r2 < 1.0 / k
    outside = r2 > 3 / 4
    r2_inside = r2[inside]
    s = 1.0 - k * r2_inside
    u_x = -2.0 * p * k * x[inside, 0] * s ** (p - 1)
    minus_lap_u = 2.0 * p * k * dim * s ** (p - 1) - 4.0 * p * (
        p - 1.0
    ) * k**2 * r2_inside * s ** (p - 2)
    f[inside] = minus_lap_u + u_x
    f = f + u_exact(x)
    f[outside] = f[outside] - 1
    return f


@torch.no_grad()
def forcing_value(x):
    
    return f_rhs(x)


@torch.no_grad()
def make_evaluation_grid(dimension: int | None = None):
    
    coordinates = torch.linspace(-1, 1, 201, device=device)
    X, Y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    if dimension is None:
        points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    else:
        points = torch.zeros(X.numel(), dimension, device=device)
        points[:, 0] = X.reshape(-1)
        points[:, 1] = Y.reshape(-1)
    exact = u_exact(points)
    exact_norm = torch.sqrt(torch.mean(exact**2))
    return X, Y, points, exact, exact_norm


@torch.no_grad()
def evaluate_solution(u_model, points, exact, exact_norm):
    
    prediction = u_model(points)
    l2 = (torch.sqrt(torch.mean((prediction - exact) ** 2)) / exact_norm).item()
    linf = torch.max(torch.abs(prediction - exact)).item()
    return prediction, l2, linf


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


def sample_interior_neri(N: int) -> torch.Tensor:
    x = 2.0 * torch.rand(N, d, device=device) - 1.0
    return x


def sample_interior(N: int) -> torch.Tensor:
    x = 2.0 * torch.rand(N, 2, device=device) - 1.0
    return x


def sample_boundary_per_face(N: int, d: int, device=device):
    x_list = []
    for j in range(d):
        pts0 = 2.0 * torch.rand(N, d, device=device) - 1.0
        pts0[:, j] = -1.0
        x_list.append(pts0)
        pts1 = 2.0 * torch.rand(N, d, device=device) - 1.0
        pts1[:, j] = 1.0
        x_list.append(pts1)
    return torch.cat(x_list, dim=0)


def outward_normal_cube(xb: torch.Tensor, tol: float = 1e-06) -> torch.Tensor:
    normal = torch.zeros_like(xb)
    mask_minus = torch.isclose(xb, -torch.ones_like(xb), atol=tol, rtol=0.0)
    mask_plus = torch.isclose(xb, torch.ones_like(xb), atol=tol, rtol=0.0)
    normal[mask_minus] = -1.0
    normal[mask_plus] = 1.0
    return normal


class ResidualBlock(nn.Module):

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.act(self.fc1(z))
        out = self.act(self.fc2(out))
        return out + z


class ResidualNetwork(nn.Module):

    def __init__(self, in_dim: int = 2, width: int = 80, blocks: int = 4):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, width)
        self.act = nn.Tanh()
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(blocks)])
        self.fcout = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.fc0(x))
        for b in self.blocks:
            z = b(z)
        return self.fcout(z).squeeze(-1)


def grad_u(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]


def neri_loss(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ux = grad_u(u, x)
    vx = ux.detach()
    v = u.detach()
    f = forcing_value(x)
    integrand = (vx * ux).sum(dim=1) + coeff_ns * vx[:, 0] * u + u * v - f * u
    return torch.mean(integrand)


def pinn_loss(
    u: torch.Tensor, x: torch.Tensor, uxb, xb, *, boundary_weight=100, tau=0.01
) -> torch.Tensor:
    ux = grad_u(u, x)
    uxx = laplace_u(u, x)
    f = forcing_value(x)
    u_1 = u - tau * (-uxx + coeff_ns * ux[:, 0] + u - f)
    u_2 = torch.relu(u_1 - 1) + 1
    rr = (u - u_2) ** 2
    nb = outward_normal_cube(xb)
    Neum = (uxb * nb).sum(dim=1)
    return torch.mean(rr) / tau + boundary_weight * torch.mean(Neum**2)


def wan_min_loss(u, v, x, *, wan_regularization=5e-4):
    ux = grad_u(u, x)
    vx = grad_u(v, x)
    v = v.detach()
    vx = vx.detach()
    w = u - v
    wx = ux - vx
    f = forcing_value(x)
    integrand = (ux * wx).sum(dim=1) + ux[:, 0] * w + u * w - f * w
    return torch.mean(integrand) - wan_regularization * torch.mean(
        w * w + (wx * wx).sum(dim=1)
    )


def wan_max_loss(u, v, x, *, wan_regularization=5e-4):
    ux = grad_u(u, x)
    vx = grad_u(v, x)
    ux = ux.detach()
    f = forcing_value(x)
    w = u - v
    wx = ux - vx
    integrand = (ux * wx).sum(dim=1) + ux[:, 0] * w + u * w - f * w
    return -(
        torch.mean(integrand)
        - wan_regularization * torch.mean(w * w + (wx * wx).sum(dim=1))
    )


def plot_solution(X, Y, u_nn, u_ex, diff, filename):
    
    import matplotlib.pyplot as plt

    Xn = X.cpu().numpy()
    Yn = Y.cpu().numpy()
    u_nn_np = u_nn.numpy()
    u_ex_np = u_ex.numpy()
    diff_np = diff.numpy()
    if filename is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(DATA_DIR / filename, Xn=Xn, Yn=Yn, u_nn=u_nn_np, uex=u_ex_np)
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(
        Xn, Yn, u_nn_np, cmap="viridis", linewidth=0, antialiased=True
    )
    ax1.set_title("NN solution $u_{\\mathrm{NN}}$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=12)
    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(
        Xn, Yn, u_ex_np, cmap="viridis", linewidth=0, antialiased=True
    )
    ax2.set_title("Exact solution $u_{\\mathrm{exact}}$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=12)
    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(
        Xn, Yn, diff_np, cmap="inferno", linewidth=0, antialiased=True
    )
    ax3.set_title("Absolute error $|u_{\\mathrm{NN}}-u_{\\mathrm{exact}}|$")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("error")
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=12)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def report_solution(X, Y, prediction, exact, filename, method):
    
    u_nn = prediction.reshape(X.shape).cpu()
    u_ex = exact.reshape(X.shape).cpu()
    diff = (u_nn - u_ex).abs()
    print(
        f"[{method} grid] mean|diff|=",
        diff.mean().item(),
        " max|diff|=",
        diff.max().item(),
    )
    plot_solution(X, Y, u_nn, u_ex, diff, filename)


def train_neri(seed: int | None = None, params: Parameters | None = None):
    if params is None:
        params = Parameters(method="neri")
    if seed is None:
        seed = params.seed
    set_seed(seed)
    z_u = ResidualNetwork(in_dim=d, width=params.width, blocks=params.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), params)
    sch_u = make_scheduler(opt_u, params, total_steps=params.epochs)

    def u_model(x):
        return z_u(x) ** 2 + 1

    X, Y, pts, uex, exact_norm = make_evaluation_grid(d)
    ute = None
    t0 = time.time()
    for ep in range(params.epochs):
        x = sample_interior_neri(params.N)
        x.requires_grad_(True)
        u = u_model(x)
        Loss = neri_loss(u, x)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u, ep + 1)
        if ep % 1000 == 0 or ep == params.epochs - 1:
            ute, l2, linf = evaluate_solution(u_model, pts, uex, exact_norm)
            dt = time.time() - t0
            print(f"[NeRI] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s")
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        report_solution(X, Y, ute, uex, "VI_NeRI.npz", "NeRI")
    return z_u


def train_pinn(seed: int | None = None, params: Parameters | None = None):
    if params is None:
        params = Parameters(method="pinn")
    if seed is None:
        seed = params.seed
    set_seed(seed)
    z_u = ResidualNetwork(in_dim=2, width=params.width, blocks=params.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), params)
    sch_u = make_scheduler(opt_u, params, total_steps=params.epochs)

    def u_model(x):
        return z_u(x) ** 2 + 1

    X, Y, pts, uex, exact_norm = make_evaluation_grid()
    ute = None
    t0 = time.time()
    for ep in range(params.epochs):
        x = sample_interior(params.N)
        x.requires_grad_(True)
        u = u_model(x)
        xb = sample_boundary_per_face(params.Nb, 2, device=device)
        xb.requires_grad_(True)
        ub = u_model(xb)
        uxb = grad_u(ub, xb)
        Loss = pinn_loss(
            u, x, uxb, xb, boundary_weight=params.boundary_weight, tau=params.tau
        )
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u, ep + 1)
        if ep % 500 == 0 or ep == params.epochs - 1:
            ute, l2, linf = evaluate_solution(u_model, pts, uex, exact_norm)
            dt = time.time() - t0
            print(f"[Prox-PINN] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s")
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        report_solution(X, Y, ute, uex, "VI_PINN.npz", "Prox-PINN")
    return z_u


def train_wan(seed: int | None = None, params: Parameters | None = None):
    if params is None:
        params = Parameters(method="wan")
    if seed is None:
        seed = params.seed
    set_seed(seed)
    z_u = ResidualNetwork(in_dim=2, width=params.width, blocks=params.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), params)
    sch_u = make_scheduler(opt_u, params, total_steps=params.epochs)
    z_v = ResidualNetwork(in_dim=2, width=params.width, blocks=params.blocks).to(device)
    opt_v = make_optimizer(z_v.parameters(), params)
    sch_v = make_scheduler(
        opt_v, params, total_steps=params.epochs, step_size=params.test_step_size
    )

    def u_model(x):
        return z_u(x) ** 2 + 1

    def v_model(x):
        return z_v(x) ** 2 + 1

    X, Y, pts, uex, exact_norm = make_evaluation_grid()
    ute = None
    t0 = time.time()
    for ep in range(params.epochs):
        for _ in range(1):
            x = sample_interior(params.N).requires_grad_(True)
            u = u_model(x)
            v = v_model(x)
            loss_v = wan_max_loss(u, v, x, wan_regularization=params.wan_regularization)
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            opt_v.step()
            step_scheduler(sch_v)
        for _ in range(1):
            x = sample_interior(params.N).requires_grad_(True)
            u = u_model(x)
            v = v_model(x)
            loss_u = wan_min_loss(u, v, x, wan_regularization=params.wan_regularization)
            opt_u.zero_grad(set_to_none=True)
            loss_u.backward()
            opt_u.step()
            step_scheduler(sch_u)
        if ep % 1000 == 0 or ep == params.epochs - 1:
            ute, l2, linf = evaluate_solution(u_model, pts, uex, exact_norm)
            dt = time.time() - t0
            print(f"[WAN] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s")
    with torch.no_grad():
        if ute is None:
            ute = u_model(pts)
        report_solution(X, Y, ute, uex, "VI_WAN.npz", "WAN")
    return (z_u, z_v)


def main(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters()
    trainers = {"neri": train_neri, "pinn": train_pinn, "wan": train_wan}
    if parameters.method not in trainers:
        raise ValueError(
            f"Unknown method: {parameters.method}. Choose neri, pinn, or wan."
        )
    print(f"Using device: {device}")
    print("Running...")
    return trainers[parameters.method](seed=parameters.seed, params=parameters)


if __name__ == "__main__":
    parameters = Parameters()
    if _args.method is not None:
        parameters.method = _args.method
    if _args.seed is not None:
        parameters.seed = _args.seed
    main(parameters)
