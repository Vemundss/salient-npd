import numpy as np
import pandas as pd
import matplotlib.patheffects as PathEffects


def histogram_equalization(x, num_colors=256):
    """
    Do histogram equalization

    Theory source:
    https://www.uio.no/studier/emner/matnat/ifi/INF2310/v20/undervisningsmateriale/forelesning/inf2310-2020-05-histogramoperasjoner.pdf
    """
    tmp = x - np.min(x)
    tmp /= np.max(tmp)  # x \in [0,1]
    tmp = np.around(tmp * (num_colors - 1))  # x \in [0,255] \subseq \mathbb{N}

    n = len(tmp)
    px = np.zeros(num_colors)
    for i in range(num_colors):
        px[i] = np.sum(tmp == i) / n

    cdfx = np.cumsum(px)
    return cdfx[tmp.astype(int)]  # [0,1]


def scattera(ax, x, y, s, c, alpha, **kwargs):
    """
    Matplotlib scatterplot with individual transparency (alpha)
    values (vector)

    PS! slow, since we have to plot every point individually
    in a loop

    PS2! This is strictly speaking unnecessary, since
    we can instead pass a 4D vector of RGBA values
    into "c". Thus, just take whatever color previously
    selected, then add a transparency channel to that color
    and pass that to the argument list to matplotlib's scatter()
    func.
    """
    try:
        x, y = x.values, y.values  # x,y is pd.Series
    except AttributeError:
        pass

    # sort wrt. scale, s.t. smalles scatter blobs are plotted last
    idx = np.argsort(s)
    x, y, s, c, alpha = (
        np.flip(x[idx]),
        np.flip(y[idx]),
        np.flip(s[idx]),
        np.flip(c[idx]),
        np.flip(alpha[idx]),
    )

    for xi, yi, si, ci, alphai in zip(x, y, s, c, alpha):
        ax.scatter(xi, yi, s=si, c=ci, alpha=alphai, **kwargs)
    return ax


def spatial_neural_activity_scatterplot(
    ax, x, y, z, alpha, xlabel, ylabel, zlabel, title, brain_region_names
):
    """
    OBS! Starts to add same color annotation text as color of its belonging data points
        - it is fix! :D

    Args:
        ax: matplotlib ax
        x,y,z: data vectors
        xlabel,ylabel,ylabel: label of axis
        title: plot title
        brain_region_names: name of brain region the data point (x,y,z) belongs to
    """

    # heuristical percetual satisfying transform
    dv = histogram_equalization(z)
    s = np.exp(5 * dv)

    unique_brain_region_names, inverse_idxs = np.unique(
        brain_region_names, return_inverse=True
    )
    # sample and normalise (unit vector) colors (RGB + a, later)
    unique_colors = np.random.uniform(0, 1, (len(brain_region_names), 4))
    unique_colors[:, :-1] = (
        unique_colors[:, :-1]
        / np.sqrt(np.sum(unique_colors[:, :-1] ** 2, axis=1))[:, None]
    )

    # Add the unique color to every data point with the same brain region
    c = np.zeros((inverse_idxs.shape[0], 4))
    ui_idxs = np.unique(inverse_idxs)
    for i, brain_region_idx in enumerate(ui_idxs):
        c[inverse_idxs == brain_region_idx] = unique_colors[i]
    c[:, -1] = alpha  # Add transparency

    ax.scatter(x, y, s=s, c=c)  # THE IT

    unique_colors[:, -1] = 1  # no transparency
    # plot annotated brain region names
    for brain_region_name, color in zip(unique_brain_region_names, unique_colors):
        idxs = brain_region_names == brain_region_name
        med_x, med_y = np.median(x[idxs]), np.median(y[idxs])
        txt = ax.annotate(brain_region_name, (med_x, med_y), color=color)
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=1, foreground="black")]
        )  # add contrast border to annotation txt

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.twinx().set_ylabel(zlabel, rotation=270, labelpad=17, color="blue")

    return ax


if __name__ == "__main__":

    # test histogram equalization
    import matplotlib.pyplot as plt

    a = np.abs(np.random.normal(loc=0, scale=1, size=100))
    a = a / np.max(a)

    plt.hist(a, cumulative=True, density=True)
    plt.hist(histogram_equalization(a), cumulative=True, density=True)
    plt.legend(["a", "hist_eq(a)"])
    plt.show()
