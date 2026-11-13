

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def data_path(filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def main():
    
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load(data_path("NeRI.npz"))
    Xn = data["Xn"]
    Yn = data["Yn"]
    uex = data["uex"]
    u_nn = data["u_nn"]
    error_history = data["error_history"]
    data2 = np.load(data_path("PINN.npz"))
    u_nn2 = data2["u_nn"]
    error_history2 = data2["error_history"]
    data3 = np.load(data_path("WAN.npz"))
    u_nn3 = data3["u_nn"]
    error_history3 = data3["error_history"]
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
    levels = np.linspace(min_all_u, max_all_u, 300)
    levels_e = np.linspace(min_all_err, max_all_err, 300)
    plt.figure(figsize=(7, 5))
    indices = np.arange(0, len(error_history), 20)
    plt.plot(indices, np.asarray(error_history)[::20], label="NeRI", linewidth=1.2)
    plt.plot(indices, np.asarray(error_history2)[::20], label="PINN", linewidth=1.2)
    plt.plot(indices, np.asarray(error_history3)[::20], label="WAN", linewidth=1.2)
    plt.xlabel("Index")
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_path("error_history_corner.png"), dpi=200)
    plt.show()
    plt.close()
    fig = plt.figure(figsize=(32, 13))
    ax = fig.add_subplot(2, 4, 1)
    contour = ax.contourf(
        Xn, Yn, uex, cmap="jet", levels=levels, vmin=min_all_u, vmax=max_all_u
    )
    ax.set_title("U_exact", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 2)
    contour = ax.contourf(
        Xn, Yn, u_nn, cmap="jet", levels=levels, vmin=min_all_u, vmax=max_all_u
    )
    ax.set_title("U_NeRI", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 3)
    contour = ax.contourf(
        Xn, Yn, u_nn2, cmap="jet", levels=levels, vmin=min_all_u, vmax=max_all_u
    )
    ax.set_title("U_PINN", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    ax = fig.add_subplot(2, 4, 4)
    contour = ax.contourf(
        Xn, Yn, u_nn3, cmap="jet", levels=levels, vmin=min_all_u, vmax=max_all_u
    )
    ax.set_title("U_WAN", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
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
    ax.set_title("Error_PINN", fontsize=18, pad=25)
    ax.set_aspect("equal", adjustable="box")
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
    plt.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)
    filename = data_path("Corner_Singular.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
