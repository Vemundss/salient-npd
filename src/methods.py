import numpy as np
import pandas as pd


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
    """
    try:
        x,y = x.values,y.values # x,y is pd.Series
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


if __name__ == "__main__":

    # test histogram equalization
    import matplotlib.pyplot as plt

    a = np.abs(np.random.normal(loc=0, scale=1, size=100))
    a = a / np.max(a)

    plt.hist(a, cumulative=True, density=True)
    plt.hist(histogram_equalization(a), cumulative=True, density=True)
    plt.legend(["a", "hist_eq(a)"])
    plt.show()
