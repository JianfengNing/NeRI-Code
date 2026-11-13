

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["neri", "pinn"],
        default=None,
        help="Override Parameters.method for this run.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override Parameters.seed."
    )
    return parser.parse_args()



if __name__ == "__main__":
    arguments = parse_arguments()

import torch
import torch.nn as nn


@dataclass
class Parameters:
    
    method: str = 'neri'  
    seed: int = 0  
    interior_points: int = 10000  
    boundary_points: int = 2000  
    boundary_weight: float = 200.0  
    
    epochs: int=6000
    width: int = 100  
    
    blocks: int =3
    activation: str = 'tanh'  
    learning_rate: float = 0.001  
    optimizer: str = 'adam'  
    
    scheduler: str = 'cosine'
    step_size: int | None = None  
    gamma: float | None = None  
    milestones: tuple[int, ...] | None = None  
    cosine_t_max: int | None = None  
    eta_min: float = 1e-8  
    weight_decay: float | None = None  
    momentum: float = 0.0  
    

    def __post_init__(self):
        if self.method not in ("neri", "pinn"):
            raise ValueError("method must be 'neri' or 'pinn'")
        if self.epochs is None:
            self.epochs = 5000 if self.method == "neri" else 10000
        if self.blocks is None:
            self.blocks = 3 if self.method == "neri" else 4
        if self.scheduler is None:
            self.scheduler = "step"
        if self.step_size is None:
            self.step_size = 200 if self.method == "neri" else 500
        if self.gamma is None:
            self.gamma = 0.8
        if self.weight_decay is None:
            self.weight_decay = 0.01 if self.optimizer == "adamw" else 0.0
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError("optimizer must be 'adam', 'adamw', or 'sgd'")
        if self.scheduler not in ("step", "multistep", "cosine", "constant"):
            raise ValueError(
                "scheduler must be 'step', 'multistep', 'cosine', or 'constant'"
            )


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dimension = 10
OMEGA_MEAS = 1.0
BDRY_MEAS = 2 * dimension
p = 1.5
EPS = 1e-10


def make_optimizer(model_parameters, parameters: Parameters):
    
    if parameters.optimizer == "adam":
        return torch.optim.Adam(
            model_parameters,
            lr=parameters.learning_rate,
            weight_decay=parameters.weight_decay,
        )
    if parameters.optimizer == "adamw":
        return torch.optim.AdamW(
            model_parameters,
            lr=parameters.learning_rate,
            weight_decay=parameters.weight_decay,
        )
    return torch.optim.SGD(
        model_parameters,
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
        momentum=parameters.momentum,
    )


def make_scheduler(
    optimizer, parameters: Parameters, *, total_steps: int, step_size=None
):
    
    if parameters.scheduler == "step":
        interval = parameters.step_size if step_size is None else step_size
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=interval, gamma=parameters.gamma
        )
    if parameters.scheduler == "multistep":
        milestones = parameters.milestones
        if milestones is None:
            milestones = (total_steps // 2, 3 * total_steps // 4)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=parameters.gamma
        )
    if parameters.scheduler == "cosine":
        horizon = (
            total_steps if parameters.cosine_t_max is None else parameters.cosine_t_max
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=parameters.eta_min
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


def step_scheduler(scheduler, epoch=None):
    
    if epoch is not None and isinstance(
        scheduler,
        (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.MultiStepLR),
    ):
        scheduler.step(epoch)
    else:
        scheduler.step()


def set_neri_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def neri_exact_solution(x: torch.Tensor) -> torch.Tensor:
    rr = torch.sum(x**2, dim=1, keepdim=False)
    uu = 1 - rr**0.5
    return uu


def neri_forcing(x: torch.Tensor) -> torch.Tensor:
    r = torch.sum(x**2, dim=1, keepdim=False) ** 0.5
    ff = (dimension - 1 - x[:, 0]) / r
    return ff


def sample_interior(N: int) -> torch.Tensor:
    x = torch.rand(N, dimension, device=device)
    return x


def sample_boundary(N, d, device=device):
    
    x_list = []
    for j in range(d):
        pts0 = torch.rand(N, d, device=device)
        pts0[:, j] = 0.0
        x_list.append(pts0)
        pts1 = torch.rand(N, d, device=device)
        pts1[:, j] = 1.0
        x_list.append(pts1)
    return torch.cat(x_list, dim=0)


def sample_boundary_per_face_with_normal(N, d, device=device):
    
    x = sample_boundary(N, d, device=device)
    n = torch.zeros_like(x)
    for j in range(d):
        n[2 * j * N : (2 * j + 1) * N, j] = -1.0
        n[(2 * j + 1) * N : (2 * j + 2) * N, j] = 1.0
    return x, n


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


def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]


def neri_loss(
    u: torch.Tensor,
    x: torch.Tensor,
    ub: torch.Tensor,
    g: torch.Tensor,
    boundary_weight: float = 50,
) -> torch.Tensor:
    ux = gradient(u, x)
    vx = ux.detach()
    with torch.no_grad():
        f = neri_forcing(x)
    a1 = (vx**2).sum(dim=1) ** ((p - 2) / 2)
    a2 = (vx * ux).sum(dim=1)
    a3 = vx[:, 0] * u
    a4 = f * u
    integrand = a1 * a2 + a3 - a4
    integrand_b = (ub - g) ** 2
    return torch.mean(integrand) + boundary_weight * BDRY_MEAS * torch.mean(integrand_b)


def neri_loss_regular(
    u: torch.Tensor,
    x: torch.Tensor,
    ub: torch.Tensor,
    g: torch.Tensor,
    boundary_weight: float = 50,
    *,
    smoothing_epsilon: float = 1e-3,
) -> torch.Tensor:
    if smoothing_epsilon <= 0:
        raise ValueError("smoothing_epsilon must be positive.")

    ux = gradient(u, x)
    vx = ux.detach()

    with torch.no_grad():
        f = neri_forcing(x)

    
    grad_sq = (vx**2).sum(dim=1) + x.shape[1] * smoothing_epsilon
    a1 = grad_sq ** ((p - 2) / 2)
    a2 = (vx * ux).sum(dim=1)
    a3 = vx[:, 0] * u
    a4 = f * u

    integrand = a1 * a2 + a3 - a4
    integrand_b = (ub - g) ** 2

    return (
        torch.mean(integrand)
        + boundary_weight * BDRY_MEAS * torch.mean(integrand_b)
    )


@torch.no_grad()
def make_slice_grid(grid_n: int = 201):
    
    xs = torch.linspace(0, 1, grid_n, device=device)
    ys = torch.linspace(0, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.zeros(X.numel(), dimension, device=device)
    pts[:, 0] = X.reshape(-1)
    pts[:, 1] = Y.reshape(-1)
    pts[:, 2:] = 0.5
    return X, Y, pts


def train_neri(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters(method="neri")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_neri_seed(parameters.seed)
    z_u = ResidualNetwork(
        in_dim=dimension, width=parameters.width, blocks=parameters.blocks
    ).to(device)
    opt_u = make_optimizer(z_u.parameters(), parameters)
    sch_u = make_scheduler(opt_u, parameters, total_steps=parameters.epochs)

    def u_model(x):
        return z_u(x)

    grid_n = 201
    X, Y, pts = make_slice_grid(grid_n)
    with torch.no_grad():
        grid_exact = neri_exact_solution(pts).reshape(grid_n, grid_n).cpu()
        grid_exact_norm = torch.mean(grid_exact**2)
    t0 = time.time()
    for ep in range(parameters.epochs):
        x = sample_interior(parameters.interior_points)
        x.requires_grad_(True)
        u = u_model(x)
        xb = sample_boundary(
            parameters.boundary_points,
            dimension,
            device=device,
        )
        with torch.no_grad():
            g = neri_exact_solution(xb).flatten()
        ub = u_model(xb)
        Loss = neri_loss(u, x, ub, g, boundary_weight=parameters.boundary_weight)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u, ep + 1)
        if ep % 500 == 0 or ep == parameters.epochs - 1:
            with torch.no_grad():
                u_e = neri_exact_solution(x).cpu()
                l2_h = torch.sqrt(
                    torch.mean((u.cpu() - u_e) ** 2) / torch.mean(u_e**2)
                ).item()
                ute = u_model(pts).reshape(grid_n, grid_n).cpu()
                l2 = torch.sqrt(
                    torch.mean((ute - grid_exact) ** 2) / grid_exact_norm
                ).item()
                linf = torch.max(torch.abs(ute - grid_exact)).item()
            dt = time.time() - t0
            print(
                f"[NeRI] ep={ep:5d}  L2={l2:.3e} L2h={l2_h:.3e}  Linf={linf:.3e} t={dt:.1f}s"
            )
    with torch.no_grad():
        u_nn = u_model(pts).reshape(grid_n, grid_n).cpu()
        u_ex = grid_exact
        diff = (u_nn - u_ex).abs()
        print(
            "[NeRI] [grid] mean|diff|=",
            diff.mean().item(),
            " max|diff|=",
            diff.max().item(),
        )
        Xn = X.cpu().numpy()
        Yn = Y.cpu().numpy()
        u_nn_np = u_nn.numpy()
        u_ex_np = u_ex.numpy()
        diff_np = diff.numpy()
        import numpy as np

        np.savez(DATA_DIR / "NeRI15.npz", Xn=Xn, Yn=Yn, u_nn=u_nn_np, uex=u_ex_np)
        plot_solution_surfaces(
            Xn, Yn, u_nn_np, u_ex_np, diff_np, DATA_DIR / "NeRI.png", pinn=False
        )
    return z_u


class PINNResidualNetwork(ResidualNetwork):

    def __init__(self, in_dim: int = 10, width: int = 80, blocks: int = 4):
        super().__init__(in_dim=in_dim, width=width, blocks=blocks)


def set_pinn_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pinn_exact_solution(x: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(torch.sum(x**2, dim=1))
    return 1.0 - r


def pinn_forcing(x: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(torch.sum(x**2, dim=1))
    return (dimension - 1 - x[:, 0]) / r


def div_flux_p_laplace(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x.requires_grad_(True)
    u = model(x)
    ux = gradient(u, x)
    grad_sq = torch.sum(ux**2, dim=1, keepdim=True) + EPS
    coeff = grad_sq ** ((p - 2.0) / 2.0)
    flux = coeff * ux
    div_flux = torch.zeros(x.shape[0], device=x.device)
    for j in range(dimension):
        flux_j = flux[:, j]
        grad_flux_j = torch.autograd.grad(
            flux_j,
            x,
            grad_outputs=torch.ones_like(flux_j),
            create_graph=True,
            retain_graph=True,
        )[0][:, j]
        div_flux = div_flux + grad_flux_j
    return (u, ux, div_flux)


def pinn_loss(
    model: nn.Module,
    x: torch.Tensor,
    xb: torch.Tensor,
    g: torch.Tensor,
    boundary_weight: float = 50.0,
) -> torch.Tensor:
    u, ux, div_flux = div_flux_p_laplace(model, x)
    convection = ux[:, 0]
    with torch.no_grad():
        f = pinn_forcing(x)
    residual = -div_flux + convection - f
    loss_pde = torch.mean(residual**2)
    ub = model(xb)
    loss_b = torch.mean((ub - g) ** 2)
    return loss_pde + boundary_weight * BDRY_MEAS * loss_b


def train_pinn(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters(method="pinn")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_pinn_seed(parameters.seed)
    z_u = PINNResidualNetwork(
        in_dim=dimension, width=parameters.width, blocks=parameters.blocks
    ).to(device)
    opt_u = make_optimizer(z_u.parameters(), parameters)
    sch_u = make_scheduler(opt_u, parameters, total_steps=parameters.epochs)

    def u_model(x):
        return z_u(x)

    grid_n = 201
    X, Y, pts = make_slice_grid(grid_n)
    with torch.no_grad():
        grid_exact = pinn_exact_solution(pts).reshape(grid_n, grid_n).cpu()
        grid_exact_norm = torch.mean(grid_exact**2)
    t0 = time.time()
    for ep in range(parameters.epochs):
        x = sample_interior(parameters.interior_points)
        x.requires_grad_(True)
        xb = sample_boundary(
            parameters.boundary_points,
            dimension,
            device=device,
        )
        with torch.no_grad():
            g = pinn_exact_solution(xb).flatten()
        Loss = pinn_loss(z_u, x, xb, g, boundary_weight=parameters.boundary_weight)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u)
        if ep % 100 == 0 or ep == parameters.epochs - 1:
            with torch.no_grad():
                u_pred_batch = u_model(x)
                u_ex_batch = pinn_exact_solution(x)
                u_pred_batch_cpu = u_pred_batch.cpu()
                u_ex_batch_cpu = u_ex_batch.cpu()
                l2_h = torch.sqrt(
                    torch.mean((u_pred_batch_cpu - u_ex_batch_cpu) ** 2)
                    / torch.mean(u_ex_batch_cpu**2)
                ).item()
                ute = u_model(pts).reshape(grid_n, grid_n).cpu()
                l2 = torch.sqrt(
                    torch.mean((ute - grid_exact) ** 2) / grid_exact_norm
                ).item()
                linf = torch.max(torch.abs(ute - grid_exact)).item()
            dt = time.time() - t0
            print(
                f"[PINN] ep={ep:5d}  loss={Loss.item():.3e}  L2={l2:.3e}  "
                f"L2h={l2_h:.3e}  Linf={linf:.3e}  t={dt:.1f}s"
            )
    with torch.no_grad():
        u_nn = u_model(pts).reshape(grid_n, grid_n).cpu()
        u_ex = grid_exact
        diff = torch.abs(u_nn - u_ex)
        print(
            "[PINN] [grid] mean|diff|=",
            diff.mean().item(),
            " max|diff|=",
            diff.max().item(),
        )
        Xn = X.cpu().numpy()
        Yn = Y.cpu().numpy()
        u_nn_np = u_nn.numpy()
        u_ex_np = u_ex.numpy()
        diff_np = diff.numpy()
        import numpy as np

        np.savez(DATA_DIR / "PINN.npz", Xn=Xn, Yn=Yn, u_nn=u_nn_np, uex=u_ex_np)
        plot_solution_surfaces(
            Xn,
            Yn,
            u_nn_np,
            u_ex_np,
            diff_np,
            DATA_DIR / "PINN_Example_5_2_ii.png",
            pinn=True,
        )
    return z_u


def plot_solution_surfaces(Xn, Yn, prediction, exact, error, filename, pinn=False):
    
    import matplotlib.pyplot as plt

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = DATA_DIR / Path(filename).name
    solution_title = (
        r"PINN solution $u_{\mathrm{PINN}}$"
        if pinn
        else r"NN solution $u_{\mathrm{NN}}$"
    )
    error_title = (
        r"Absolute error $|u_{\mathrm{PINN}}-u_{\mathrm{exact}}|$"
        if pinn
        else r"Absolute error $|u_{\mathrm{NN}}-u_{\mathrm{exact}}|$"
    )
    panels = (
        (prediction, solution_title, "viridis", "u"),
        (exact, r"Exact solution $u_{\mathrm{exact}}$", "viridis", "u"),
        (error, error_title, "inferno", "error"),
    )
    fig = plt.figure(figsize=(18, 5))
    for index, (values, title, cmap, label) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        surface = ax.plot_surface(
            Xn, Yn, values, cmap=cmap, linewidth=0, antialiased=True
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(label)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=12)
    plt.tight_layout()
    if pinn:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
    else:
        plt.savefig(filename)
    plt.show()


def main(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters()
    if parameters.method not in ("neri", "pinn"):
        raise ValueError(f"Unknown method: {parameters.method}")
    method_name = "NeRI" if parameters.method == "neri" else "PINN"
    print(f"[{method_name}] Using device: {device}")
    if parameters.method == "neri":
        return train_neri(parameters)
    return train_pinn(parameters)


if __name__ == "__main__":
    overrides = {
        name: value for name, value in vars(arguments).items() if value is not None
    }
    main(Parameters(**overrides))
