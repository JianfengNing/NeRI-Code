

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load(DATA_DIR / "VI_NeRI.npz")
    Xn = data["Xn"]
    Yn = data["Yn"]
    uex = data["uex"]
    u_nn = data["u_nn"]
    data2 = np.load(DATA_DIR / "VI_PINN.npz")
    u_nn2 = data2["u_nn"]
    data3 = np.load(DATA_DIR / "VI_WAN.npz")
    u_nn3 = data3["u_nn"]
    rel_NeRI = np.sqrt(np.mean((uex - u_nn) ** 2) / np.mean(uex**2))
    rel_PINN = np.sqrt(np.mean((uex - u_nn2) ** 2) / np.mean(uex**2))
    rel_WAN = np.sqrt(np.mean((uex - u_nn3) ** 2) / np.mean(uex**2))
    print(
        f"[relative L2 error]  NeRI={rel_NeRI:.3e}  PINN={rel_PINN:.3e} WAN={rel_WAN:.3e}"
    )
    max_all_u = max(np.max(u_nn), np.max(u_nn2), np.max(u_nn3), np.max(uex))
    min_all_u = min(np.min(u_nn), np.min(u_nn2), np.min(u_nn3), np.min(uex))
    max_all_err = max(
        np.max(np.abs(u_nn - uex)),
        np.max(np.abs(u_nn2 - uex)),
        np.max(np.abs(u_nn3 - uex)),
    )
    min_all_err = 0
    levels_e = np.linspace(min_all_err, max_all_err, 300)
    fig = plt.figure(figsize=(32, 13))
    ax = fig.add_subplot(2, 4, 1, projection="3d")
    surf1 = ax.plot_surface(
        Xn,
        Yn,
        uex,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        rstride=3,
        cstride=3,
        vmin=min_all_u,
        vmax=max_all_u,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zticks([])
    ax.set_title("U_exact", fontsize=18, pad=25)
    fig.colorbar(surf1, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 2, projection="3d")
    surf1 = ax.plot_surface(
        Xn,
        Yn,
        u_nn,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        rstride=3,
        cstride=3,
        vmin=min_all_u,
        vmax=max_all_u,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zticks([])
    ax.set_title("U_NeRI", fontsize=18, pad=25)
    fig.colorbar(surf1, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 3, projection="3d")
    surf1 = ax.plot_surface(
        Xn,
        Yn,
        u_nn2,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        rstride=3,
        cstride=3,
        vmin=min_all_u,
        vmax=max_all_u,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zticks([])
    ax.set_title("U_Prox-PINN", fontsize=18, pad=25)
    fig.colorbar(surf1, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 4, projection="3d")
    surf1 = ax.plot_surface(
        Xn,
        Yn,
        u_nn3,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        rstride=3,
        cstride=3,
        vmin=min_all_u,
        vmax=max_all_u,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zticks([])
    ax.set_title("U_WAN", fontsize=18, pad=25)
    fig.colorbar(surf1, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 6)
    contour = ax.contourf(
        Xn,
        Yn,
        np.abs(u_nn - uex),
        cmap="jet",
        levels=levels_e,
        vmin=min_all_err,
        vmax=max_all_err,
    )
    ax.set_title("Error_NeRI", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax = fig.add_subplot(2, 4, 7)
    contour = ax.contourf(
        Xn,
        Yn,
        np.abs(u_nn2 - uex),
        cmap="jet",
        levels=levels_e,
        vmin=min_all_err,
        vmax=max_all_err,
    )
    ax.set_title("Error_Prox-PINN", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 8)
    contour = ax.contourf(
        Xn,
        Yn,
        np.abs(u_nn3 - uex),
        cmap="jet",
        levels=levels_e,
        vmin=min_all_err,
        vmax=max_all_err,
    )
    ax.set_title("Error_WAN", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    filename = DATA_DIR / "Variational_Inequality.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
