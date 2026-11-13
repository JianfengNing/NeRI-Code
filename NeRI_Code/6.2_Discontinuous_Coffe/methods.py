





from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["neri", "pinn", "wan", "fem"], default=None
    )
    parser.add_argument(
        "--reference",
        dest="reference_path",
        default=None,
        help="FEM reference archive; relative paths use this example data directory.",
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
    

    
    method: str = 'wan'  
    seed: int = 0  
    N: int = 10000  
    Nb: int = 256  
    epochs: int = 20000
    epochs_wan_test: int = 3
    width: int = 100  
    blocks: int = 2  
    act: str = 'tanh'  
    learning_rate: float = 0.001  
    optimizer: str | None = None  
    
    scheduler: str = 'cosine'
    step_size: int | None = None  
    test_step_size: int = 1000  
    gamma: float = 0.8  
    cosine_t_max: int | None = None  
    eta_min: float = 1e-8  
    weight_decay: float | None = None  
    momentum: float = 0.0  
    reference_path: str = 'solution.npz'  
    fem_n: int = 300  
    

    def __post_init__(self):
        if self.method not in ("neri", "pinn", "wan", "fem"):
            raise ValueError(f"Unknown method: {self.method}")
        if self.optimizer is None:
            self.optimizer = "adam"
        if self.scheduler is None:
            self.scheduler = "step"
        if self.step_size is None:
            self.step_size = 1000 if self.method == "wan" else 500
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
OMEGA_MEAS = 1.0
BDRY_MEAS = 4.0
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


def coeff_a(x):
    x_1 = x[:, 0]
    return torch.where(
        (x_1 > 2.0 / 3.0) | (x_1 < 1.0 / 3.0),
        torch.ones_like(x_1),
        3.0 * torch.ones_like(x_1),
    )


def f_rhs(x: torch.Tensor):
    xx = x[:, 0]
    return 10*torch.ones_like(xx)


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


def sample_interior(N: int) -> torch.Tensor:
    x = torch.rand(N, 2, device=device)
    return x


def eta(x: torch.Tensor) -> torch.Tensor:
    return (1 - x[:, 0]) * x[:, 0] * (1 - x[:, 1]) * x[:, 1]


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


def loss_neri(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    
    ux = grad_u(u, x)
    aa = coeff_a(x)
    vx = ux.detach()
    f = f_rhs(x).detach()
    integrand = aa * (vx * ux).sum(dim=1) + vx[:, 0] * u - f * u
    return OMEGA_MEAS * torch.mean(integrand)


def loss_pinn(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    
    ux = grad_u(u, x)
    uxx = laplace_u(u, x)
    aa = coeff_a(x)
    f = f_rhs(x).detach()
    rr = (-aa * uxx + ux[:, 0] - f) ** 2
    return OMEGA_MEAS * torch.mean(rr)


def loss_wan_solution(u, v, x):
    
    ux = grad_u(u, x)
    aa = coeff_a(x)
    vx = grad_u(v, x)
    v = v.detach()
    vx = vx.detach()
    f = f_rhs(x).detach()
    integrand = aa * (ux * vx).sum(dim=1) + ux[:, 0] * v - f * v
    return torch.log(torch.mean(integrand) ** 2)
    


def loss_wan_test(u, v, x):
    
    ux = grad_u(u, x)
    aa = coeff_a(x)
    vx = grad_u(v, x)
    ux = ux.detach()
    f = f_rhs(x).detach()
    integrand = aa * (ux * vx).sum(dim=1) + ux[:, 0] * v - f * v
    return -torch.log(torch.mean(integrand) ** 2) + torch.log(
         torch.mean((vx * vx).sum(dim=1))+ WAN_EPS)
    


def plot_comparison(Xn, Yn, u_nn_np, u_ex_np, diff_np):
    
    import matplotlib.pyplot as plt

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


def load_reference(reference_path, n):
    
    path = data_path(reference_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run --method fem for the original 201 x 201 reference; NeRI requires a separate 301 x 301 reference."
        )
    data = np.load(path)
    nodes = data["nodes"]
    U = data["U"]
    if U.size != n * n:
        raise ValueError(
            f"{path} has {U.size} values; this method requires {n} x {n} = {n * n}. Supply a matching archive with --reference; the original grids are preserved."
        )
    U = torch.tensor(U, dtype=torch.float32, device=device)
    U = U.reshape(n, n)
    U = U.T.reshape(-1)
    return U


def train_neri(seed: int | None = None, params=None, reference_path=None):
    
    P = Parameters(method="neri") if params is None else params
    seed = P.seed if seed is None else seed
    reference_path = P.reference_path if reference_path is None else reference_path
    n = 301
    U = load_reference(reference_path, n)
    set_seed(seed)
    z_u = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)

    def u_model(x):
        return eta(x) * z_u(x)

    grid_n = 301
    xs = torch.linspace(0, 1, grid_n, device=device)
    ys = torch.linspace(0, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        x = sample_interior(P.N)
        x.requires_grad_(True)
        u = u_model(x)
        Loss = loss_neri(u, x)
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


def train_pinn(seed: int | None = None, params=None, reference_path=None):
    
    P = Parameters(method="pinn") if params is None else params
    seed = P.seed if seed is None else seed
    reference_path = P.reference_path if reference_path is None else reference_path
    n = 301
    U = load_reference(reference_path, n)
    set_seed(seed)
    z_u = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)

    def u_model(x):
        return eta(x) * z_u(x)

    grid_n = 301
    xs = torch.linspace(0, 1, grid_n, device=device)
    ys = torch.linspace(0, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        x = sample_interior(P.N)
        x.requires_grad_(True)
        u = u_model(x)
        Loss = loss_pinn(u, x)
        opt_u.zero_grad(set_to_none=True)
        Loss.backward()
        opt_u.step()
        step_scheduler(sch_u, ep + 1)
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


def train_wan(seed: int | None = None, params=None, reference_path=None):
    
    P = Parameters(method="wan") if params is None else params
    seed = P.seed if seed is None else seed
    reference_path = P.reference_path if reference_path is None else reference_path
    n = 301
    U = load_reference(reference_path, n)
    set_seed(seed)
    z_u = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    z_v = FDRR(in_dim=2, width=P.width, blocks=P.blocks).to(device)
    opt_u = make_optimizer(z_u.parameters(), P)
    sch_u = make_scheduler(opt_u, P, total_steps=P.epochs)
    opt_v = make_optimizer(z_v.parameters(), P)
    sch_v = make_scheduler(opt_v, P, total_steps=P.epochs_wan_test*P.epochs, step_size=P.test_step_size)

    def u_model(x):
        return eta(x) * z_u(x)

    def v_model(x):
        return eta(x) * z_v(x)

    grid_n = n
    xs = torch.linspace(0, 1, grid_n, device=device)
    ys = torch.linspace(0, 1, grid_n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    error_history = []
    with torch.no_grad():
        reference_mean_square = torch.mean(U**2)
        reference_norm = torch.sqrt(reference_mean_square)
    t0 = time.time()
    for ep in range(P.epochs):
        for _ in range(P.epochs_wan_test):
            x = sample_interior(P.N).requires_grad_(True)
            u = u_model(x)
            v = v_model(x)
            loss_v = loss_wan_test(u, v, x)
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(z_v.parameters(), 10.0)
            opt_v.step()
            step_scheduler(sch_v)
        for _ in range(1):
            x = sample_interior(P.N).requires_grad_(True)
            u = u_model(x)
            v = v_model(x)
            loss_u = loss_wan_solution(u, v, x)
            opt_u.zero_grad(set_to_none=True)
            loss_u.backward()
            torch.nn.utils.clip_grad_norm_(z_u.parameters(), 10.0)
            opt_u.step()
            step_scheduler(sch_u)
        with torch.no_grad():
            if ep % 1000 == 0 or ep == P.epochs - 1:
                ute = u_model(pts)
                error_mean_square = torch.mean((ute - U) ** 2)
                l2error = torch.sqrt(error_mean_square / reference_mean_square)
                error_history.append(l2error.item())
                l2 = (torch.sqrt(error_mean_square) / reference_norm).item()
                linf = torch.max(torch.abs(ute - U)).item()
                dt = time.time() - t0
                print(
                    f"[{method_label(P.method)}] ep={ep:5d}  L2={l2:.3e}  Linf={linf:.3e}  t={dt:.1f}s"
                )
    with torch.no_grad():
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


def generate_unit_square_mesh(n):
    h = 1.0 / n
    nodes = []
    node_id = {}
    count = 0
    for j in range(n + 1):
        for i in range(n + 1):
            x = i * h
            y = j * h
            nodes.append([x, y])
            node_id[i, j] = count
            count += 1
    nodes = np.array(nodes)
    elements = []
    for j in range(n):
        for i in range(n):
            v00 = node_id[i, j]
            v10 = node_id[i + 1, j]
            v01 = node_id[i, j + 1]
            v11 = node_id[i + 1, j + 1]
            elements.append([v00, v10, v11])
            elements.append([v00, v11, v01])
    elements = np.array(elements, dtype=int)
    boundary_nodes = []
    for j in range(n + 1):
        for i in range(n + 1):
            if i == 0 or i == n or j == 0 or (j == n):
                boundary_nodes.append(node_id[i, j])
    boundary_nodes = np.array(boundary_nodes, dtype=int)
    return (nodes, elements, boundary_nodes)


def local_matrices(coords, aK, fK):
    
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    beta = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
    gamma = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)
    Ke = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Ke[i, j] = aK * area * (beta[i] * beta[j] + gamma[i] * gamma[j])
    Ce = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Ce[i, j] = beta[j] * area / 3.0
    Fe = np.ones(3) * fK * area / 3.0
    return (Ke, Ce, Fe)


def solve_fem(n, a_func, f_func):
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    
    nodes, elements, boundary_nodes = generate_unit_square_mesh(n)
    num_nodes = nodes.shape[0]
    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)
    for elem in elements:
        coords = nodes[elem]
        centroid = np.mean(coords, axis=0)
        xc, yc = centroid
        aK = a_func(xc, yc)
        fK = f_func(xc, yc)
        Ke, Ce, Fe = local_matrices(coords, aK, fK)
        Ae = Ke + Ce
        for i_local in range(3):
            I = elem[i_local]
            b[I] += Fe[i_local]
            for j_local in range(3):
                J = elem[j_local]
                A[I, J] += Ae[i_local, j_local]
    boundary_set = set(boundary_nodes.tolist())
    interior_nodes = np.array(
        [i for i in range(num_nodes) if i not in boundary_set], dtype=int
    )
    A_int = A[interior_nodes, :][:, interior_nodes].tocsr()
    b_int = b[interior_nodes]
    U = np.zeros(num_nodes)
    U_int = spsolve(A_int, b_int)
    U[interior_nodes] = U_int
    return (nodes, elements, U)


def fem_exact_u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def fem_coeff_a(x, y):
    if x < 1.0 / 3.0 or x > 2.0 / 3.0:
        return 1.0
    else:
        return 3.0


def fem_rhs(x, y):
    return 10*np.ones_like(x)


def compute_errors(nodes, elements, U):
    U_exact_nodes = np.array([fem_exact_u(x, y) for x, y in nodes])
    linf_error = np.max(np.abs(U - U_exact_nodes))
    l2_error_sq = 0.0
    for elem in elements:
        coords = nodes[elem]
        Uh_vals = U[elem]
        centroid = np.mean(coords, axis=0)
        xc, yc = centroid
        Uh_c = np.mean(Uh_vals)
        Ue_c = fem_exact_u(xc, yc)
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        l2_error_sq += area * (Uh_c - Ue_c) ** 2
    l2_error = np.sqrt(l2_error_sq)
    return (linf_error, l2_error)


def run_fem(parameters=None):
    P = Parameters(method="fem") if parameters is None else parameters
    print(" n        h              L_inf error        L2 error")
    print("--------------------------------------------------------")
    n = P.fem_n
    h = 1.0 / n
    nodes, elements, U = solve_fem(n, a_func=fem_coeff_a, f_func=fem_rhs)
    linf, l2 = compute_errors(nodes, elements, U)
    print(f"{n:3d}   {h:.5e}   {linf:.5e}     {l2:.5e}")
    np.savez(data_path("solution.npz"), nodes=nodes, elements=elements, U=U)


def main(parameters: Parameters | None = None):
    parameters = Parameters() if parameters is None else parameters
    if parameters.method not in ("neri", "pinn", "wan", "fem"):
        raise ValueError(f"Unknown method: {parameters.method}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if parameters.method == "fem":
        return run_fem(parameters)
    print("Device:", device)
    print("Running...")
    methods = {"neri": train_neri, "pinn": train_pinn, "wan": train_wan}
    return methods[parameters.method](
        seed=parameters.seed,
        params=parameters,
        reference_path=parameters.reference_path,
    )


if __name__ == "__main__":
    overrides = {
        name: value for name, value in vars(_args).items() if value is not None
    }
    main(Parameters(**overrides))
