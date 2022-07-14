import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import PurePath

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


def