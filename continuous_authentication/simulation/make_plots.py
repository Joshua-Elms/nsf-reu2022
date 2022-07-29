import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

def format_plots():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 26
    CHONK_SIZE = 32
    font = {"family": "DIN Condensed", "weight": "bold", "size": SMALL_SIZE}
    plt.rc("font", **font)
    plt.rc("axes", titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:white")
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc(
        "figure", titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black"
    )  #  powder blue


if __name__ == "__main__":
    format_plots()
    xpoints = (1, 2)
    ypoints = (1, 2)
    fig, ax = plt.subplots()
    ax.plot(
        xpoints,
        ypoints,
        linewidth=1,
        linestyle="dashed",     
        marker="o",
        color="green",
        markersize=18,
        # fillstyle="none",
        markeredgewidth=2,
        markeredgecolor="black"
        )

    ax.add_artist(lines.Line2D([1, 1], [1, 2], linestyle="dashed"))
    ax.add_artist(lines.Line2D([1, 2], [2, 2], linestyle="dashed"))
    ax.add_artist(lines.Line2D([1, 1], [2, 2], linestyle="dashed"))

    plt.xlim(0.75, 2.25)
    plt.ylim(0.75, 2.25)
    plt.gca().set_aspect('equal', adjustable='box')

    ax.set_xticks((1, 2))
    ax.set_yticks((1, 2))

    ax.text(1.3, 2.05, "Manhattan: 2 units")
    ax.text(1.3, 1.2, "Euclidean: 1.414 units")

    fig.suptitle("Comparing Distance Metrics")

    fig.set_size_inches(w=4, h=4.2)

    plt.savefig("/Users/joshuaelms/Desktop/github_repos/nsf-reu2022/continuous_authentication/simulation/plot_dists.png", dpi=400)