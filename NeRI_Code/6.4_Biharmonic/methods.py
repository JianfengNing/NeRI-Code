

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["neri", "deep-ritz"],
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
    seed: int | None = None  
    width: int | None = None  
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
    interior_points: int = 10000  
    boundary_points: int = 2000  
    epochs: int = 5000  
    blocks: int = 3  
    activation: str = 'tanh'  
    dim: int = 10  
    depth: int = 4  
    batch_size: int = 10000  
    iterations: int = 10000  
    print_every: int = 500  
    test_size: int = 20000  
    device: torch.device = torch.device(  
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype: torch.dtype = torch.float32  
    

    def __post_init__(self):
        if self.method not in ("neri", "deep-ritz"):
            raise ValueError("method must be 'neri' or 'deep-ritz'")
        if self.seed is None:
            self.seed = 0 if self.method == "neri" else 1234
        if self.width is None:
            self.width = 100 if self.method == "neri" else 128
        if self.scheduler is None:
            self.scheduler = "step" if self.method == "neri" else "multistep"
        if self.step_size is None:
            self.step_size = 200
        if self.gamma is None:
            self.gamma = 0.8 if self.method == "neri" else 0.2
        if self.milestones is None and self.method == "deep-ritz":
            self.milestones = (self.iterations // 2, 3 * self.iterations // 4)
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


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def neri_exact_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.prod(torch.sin(math.pi * x), dim=1)


def neri_forcing(x: torch.Tensor) -> torch.Tensor:
    sin_x = torch.sin(math.pi * x)
    cos_x = torch.cos(math.pi * x)
    u = torch.prod(sin_x, dim=1)
    biharmonic_u = dimension**2 * math.pi**4 * u
    ux1 = math.pi * cos_x[:, 0] * torch.prod(sin_x[:, 1:], dim=1)
    ux2 = math.pi * cos_x[:, 1] * sin_x[:, 0] * torch.prod(sin_x[:, 2:], dim=1)
    return biharmonic_u + ux1 + ux2


def sample_neri_interior(N: int) -> torch.Tensor:
    x = torch.rand(N, dimension, device=device)
    return x


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


def neri_laplacian(
    u: torch.Tensor, x: torch.Tensor, grad: torch.Tensor | None = None
) -> torch.Tensor:
    if grad is None:
        grad = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(x.shape[1]):
        grad_i = grad[:, i]
        second = torch.autograd.grad(
            grad_i, x, torch.ones_like(grad_i), create_graph=True
        )[0][:, i]
        lap = lap + second
    return lap


def neri_loss(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ux = gradient(u, x)
    u_laplace = neri_laplacian(u, x, grad=ux)
    v_laplace = u_laplace.detach()
    vx = ux.detach()
    with torch.no_grad():
        f = neri_forcing(x)
    a1 = u_laplace * v_laplace
    a2 = (vx[:, 0] + vx[:, 1]) * u
    a3 = f * u
    integrand = a1 + a2 - a3
    return torch.mean(integrand)


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
    set_seed(parameters.seed)
    z_u = ResidualNetwork(
        in_dim=dimension, width=parameters.width, blocks=parameters.blocks
    ).to(device)
    opt_u = make_optimizer(z_u.parameters(), parameters)
    sch_u = make_scheduler(opt_u, parameters, total_steps=parameters.epochs)

    def boundary_factor(x):
        factors = 4 * x * (1.0 - x)
        return torch.prod(factors, dim=1)

    def u_model(x):
        return z_u(x) * boundary_factor(x)

    grid_n = 201
    X, Y, pts = make_slice_grid(grid_n)
    with torch.no_grad():
        grid_exact = neri_exact_solution(pts).reshape(grid_n, grid_n).cpu()
        grid_exact_norm = torch.mean(grid_exact**2)
    t0 = time.time()
    for ep in range(parameters.epochs):
        x = sample_neri_interior(parameters.interior_points)
        x.requires_grad_(True)
        u = u_model(x)
        Loss = neri_loss(u, x)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u)
        if ep % 100 == 0 or ep == parameters.epochs - 1:
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

        np.savez(DATA_DIR / "NeRI.npz", Xn=Xn, Yn=Yn, u_nn=u_nn_np, uex=u_ex_np)
        plot_solution_surfaces(
            Xn, Yn, u_nn_np, u_ex_np, diff_np, DATA_DIR / "NeRI.png", pinn=False
        )
    return z_u


class MLP(nn.Module):

    def __init__(self, dim: int, width: int, depth: int):
        super().__init__()
        layers = [nn.Linear(dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepRitzBiharmonic(nn.Module):
    

    def __init__(self, dim: int, width: int, depth: int):
        super().__init__()
        self.mlp = MLP(dim, width, depth)

    @staticmethod
    def boundary_factor(x: torch.Tensor) -> torch.Tensor:
        factors = 4.0 * x * (1.0 - x)
        return torch.prod(factors, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_factor(x) * self.mlp(x)


def ritz_exact_solution(x: torch.Tensor) -> torch.Tensor:
    
    return torch.prod(torch.sin(math.pi * x), dim=1, keepdim=True)


def ritz_forcing(x: torch.Tensor, dim: int) -> torch.Tensor:
    
    coefficient = dim**2 * math.pi**4
    return coefficient * ritz_exact_solution(x)


def ritz_laplacian(
    u: torch.Tensor, x: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        grad_u_i = grad_u[:, i : i + 1]
        second_derivative = torch.autograd.grad(
            outputs=grad_u_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_u_i),
            create_graph=create_graph,
            retain_graph=True,
        )[0][:, i : i + 1]
        lap = lap + second_derivative
    return lap


def sample_ritz_interior(
    number_of_points: int,
    dim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.rand(number_of_points, dim, device=device, dtype=dtype)
    x.requires_grad_(True)
    return x


def sample_boundary(
    number_of_points: int,
    dim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    
    x = torch.rand(number_of_points, dim, device=device, dtype=dtype)
    face_coordinate = torch.randint(
        low=0, high=dim, size=(number_of_points,), device=device
    )
    face_side = torch.randint(
        low=0,
        high=2,
        size=(number_of_points,),
        device=device,
        dtype=torch.int64,
    ).to(dtype)
    rows = torch.arange(number_of_points, device=device)
    x[rows, face_coordinate] = face_side
    x.requires_grad_(True)
    return x


def deep_ritz_loss(model: nn.Module, x: torch.Tensor, dim: int):
    
    u = model(x)
    delta_u = ritz_laplacian(u, x, create_graph=True)
    f = ritz_forcing(x, dim)
    energy_density = 0.5 * delta_u.square() - f * u
    loss = energy_density.mean()
    return (loss, u, delta_u)


@torch.no_grad()
def relative_l2_error(
    model: nn.Module,
    number_of_points: int,
    dim: int,
    device: str,
    dtype: torch.dtype,
) -> float:
    x = torch.rand(number_of_points, dim, device=device, dtype=dtype)
    prediction = model(x)
    target = ritz_exact_solution(x)
    numerator = torch.mean((prediction - target).square())
    denominator = torch.mean(target.square())
    return torch.sqrt(numerator / denominator).item()


def natural_boundary_error(
    model: nn.Module,
    number_of_points: int,
    dim: int,
    device: str,
    dtype: torch.dtype,
) -> float:
    
    x_boundary = sample_boundary(number_of_points, dim, device, dtype)
    u_boundary = model(x_boundary)
    delta_u_boundary = ritz_laplacian(u_boundary, x_boundary, create_graph=False)
    return torch.sqrt(torch.mean(delta_u_boundary.detach().square())).item()


def train_deep_ritz(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters(method="deep-ritz")
    model = DeepRitzBiharmonic(
        dim=parameters.dim,
        width=parameters.width,
        depth=parameters.depth,
    ).to(device=parameters.device, dtype=parameters.dtype)
    optimizer = make_optimizer(model.parameters(), parameters)
    scheduler = make_scheduler(optimizer, parameters, total_steps=parameters.iterations)
    start_time = time.time()
    for iteration in range(1, parameters.iterations + 1):
        x = sample_ritz_interior(
            parameters.batch_size, parameters.dim, parameters.device, parameters.dtype
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = deep_ritz_loss(model, x, parameters.dim)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at iteration {iteration}: {loss.item()}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        step_scheduler(scheduler)
        if iteration == 1 or iteration % parameters.print_every == 0:
            error = relative_l2_error(
                model,
                min(parameters.test_size, 10000),
                parameters.dim,
                parameters.device,
                parameters.dtype,
            )
            elapsed = time.time() - start_time
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"[Deep Ritz] iteration={iteration:6d} | energy={loss.item(): .6e} | "
                f"relative_L2={error: .4e} | lr={learning_rate:.2e} | time={elapsed:.1f}s"
            )
    return model


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


def run_deep_ritz(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters(method="deep-ritz")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(parameters.seed)
    if parameters.device == "cuda":
        torch.cuda.manual_seed_all(parameters.seed)
    print(f"[Deep Ritz] Device: {parameters.device}")
    print(f"[Deep Ritz] Dimension: {parameters.dim}")
    trained_model = train_deep_ritz(parameters)
    final_l2_error = relative_l2_error(
        trained_model,
        parameters.test_size,
        parameters.dim,
        parameters.device,
        parameters.dtype,
    )
    delta_u_boundary_error = natural_boundary_error(
        trained_model,
        number_of_points=4096,
        dim=parameters.dim,
        device=parameters.device,
        dtype=parameters.dtype,
    )
    print("\n[Deep Ritz] Final diagnostics")
    print("-----------------")
    print(f"[Deep Ritz] Relative L2 error: {final_l2_error:.6e}")
    print(
        "[Deep Ritz] Boundary RMS of Delta u "
        f"(not used in training): {delta_u_boundary_error:.6e}"
    )
    torch.save(
        trained_model.state_dict(),
        DATA_DIR / f"deep_ritz_biharmonic_d{parameters.dim}.pt",
    )
    return trained_model


def main(parameters: Parameters | None = None):
    if parameters is None:
        parameters = Parameters()
    if parameters.method not in ("neri", "deep-ritz"):
        raise ValueError(f"Unknown method: {parameters.method}")
    method_name = "NeRI" if parameters.method == "neri" else "Deep Ritz"
    print(f"[{method_name}] Using device: {device}")
    if parameters.method == "neri":
        return train_neri(parameters)
    return run_deep_ritz(parameters)


if __name__ == "__main__":
    overrides = {
        name: value for name, value in vars(arguments).items() if value is not None
    }
    main(Parameters(**overrides))
