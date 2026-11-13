

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_DIR / "NeRI.npz")
    Xn = data["Xn"]
    Yn = data["Yn"]
    uex = data["uex"]
    u_nn = data["u_nn"]
    max_all_u = max(np.max(u_nn), np.max(uex), 1)
    min_all_u = min(np.min(u_nn), np.min(uex), 0)
    rel_NeRI = np.sqrt(np.mean((uex - u_nn) ** 2) / np.mean(uex**2))
    max_all_err = np.max(np.abs(u_nn - uex))
    print(f"[NeRI] relative L2 error={rel_NeRI:.3e}")
    print(f"[NeRI] max_all_err={max_all_err:.3e}")
    levels = np.linspace(min_all_u, max_all_u, 300)
    fig = plt.figure(figsize=(24, 6))
    ax = fig.add_subplot(1, 3, 1)
    contour = ax.contourf(
        Xn, Yn, uex, cmap="jet", levels=levels, vmin=min_all_u, vmax=max_all_u
    )
    ax.set_title("U_exact", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(1, 3, 2)
    contour = ax.contourf(
        Xn, Yn, u_nn, cmap="jet", levels=levels, vmin=min_all_u, vmax=max_all_u
    )
    ax.set_title("U_NeRI", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(1, 3, 3)
    contour = ax.contourf(Xn, Yn, np.abs(u_nn - uex), cmap="jet", levels=300)
    ax.set_title("Error_NeRI", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    filename = DATA_DIR / "biharmonic.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
