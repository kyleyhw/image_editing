import matplotlib.pyplot as plt


def set_size(w: float, h: float, ax: plt.Axes | None = None) -> None:
    """Resize the parent figure so that ``ax`` has exact width/height in inches."""
    if ax is None:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    figw = float(w) / (right - left)
    figh = float(h) / (top - bottom)
    ax.figure.set_size_inches(figw, figh)


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot([1, 3, 2])

    set_size(5, 2, ax=axs[0, 0])

    plt.show()
