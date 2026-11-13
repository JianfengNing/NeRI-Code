from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def data_path(filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def main():
    import numpy as np
    import matplotlib.pyplot as plt

    widths = (25, 50, 100, 200)
    results = {}
    for width in widths:
        with np.load(data_path(f"NeRI_GD_{width}.npz")) as data:
            results[width] = {
                key: data[key]
                for key in ("u_nn", "error_history")
            }
            if width == 50:
                # Keep the reference used by the original plotting script.
                Xn = data["Xn"]
                Yn = data["Yn"]
                uex = data["uex"]
    errors = {
        width: np.abs(results[width]["u_nn"] - uex)
        for width in widths
    }
    relative_errors = {
        width: np.sqrt(
            np.mean((uex - results[width]["u_nn"]) ** 2) / np.mean(uex**2)
        )
        for width in widths
    }
    print(
        "[relative L2 error]  "
        + "  ".join(
            f"NeRI{width}={relative_errors[width]:.3e}" for width in widths
        )
    )

    all_solutions = [uex] + [results[width]["u_nn"] for width in widths]
    max_all_u = max(np.max(values) for values in all_solutions)
    min_all_u = min(np.min(values) for values in all_solutions)
    max_all_err = max(np.max(errors[width]) for width in widths)
    min_all_err = 0
    levels = np.linspace(min_all_u, max_all_u, 300)
    levels_e = np.linspace(min_all_err, max_all_err, 300)

    plt.figure(figsize=(7, 5))
    for width in widths:
        history = np.asarray(results[width]["error_history"])
        indices = np.arange(0, len(history), 20)
        plt.plot(indices, history[::20], label=f"NeRI{width}", linewidth=1.2)
    plt.xlabel("Index")
    plt.ylabel("Relative L2 Error")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_path("error_history_corner.png"), dpi=200)
    plt.show()
    plt.close()

    # Top row: exact solution and widths 25, 50, 100, 200.
    # Bottom row: the four corresponding absolute-error plots.
    fig, axes = plt.subplots(2, 5, figsize=(40, 13))
    panels = [(uex, "U_exact")] + [
        (results[width]["u_nn"], f"NeRI width {width}") for width in widths
    ]
    for ax, (values, title) in zip(axes[0], panels):
        contour = ax.contourf(
            Xn, Yn, values, cmap="jet", levels=levels,
            vmin=min_all_u, vmax=max_all_u,
        )
        ax.set_title(title, fontsize=18, pad=25)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)

    axes[1, 0].axis("off")
    for ax, width in zip(axes[1, 1:], widths):
        contour = ax.contourf(
            Xn, Yn, errors[width], cmap="jet", levels=levels_e,
            vmin=min_all_err, vmax=max_all_err,
        )
        ax.set_title(f"Error: NeRI width {width}", fontsize=18, pad=25)
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=ax, shrink=1.0, fraction=0.046)

    fig.savefig(data_path("Corner_Singular.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
