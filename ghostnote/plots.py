import matplotlib.pyplot as plt

from ghostnote import utils


def plot_lags_2D(
    mic_a: tuple[int, int],
    mic_b: tuple[int, int],
    d: int = utils.DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    c: float = 343.0,
    labels=["Mic A", "Mic B"],
    ax=None,
    figsize=(6, 6),
    add_colorbar=True,
    add_legend=True,
):
    """Plot lag map for 2D mic locations.

    :param mic_a: location of microphone A, in cartesian coordinates
    :param mic_b: location of microphone A, in cartesian coordinates
    :param d: diameter of the drum, in centimeters
    :param sr: sampling rate
    :param scale: scale to increase/decrease precision originally in
        centimeters.  For example, for millimeters, scale should be 10
    :param medium: the medium the sound travels through.  One of 'air' or
        'drumhead', the latter for optical/magnetic measurements
    """
    r = d * scale / 2
    mic_a = utils.polar_to_cartesian(mic_a[0] * r, mic_a[1])
    mic_b = utils.polar_to_cartesian(mic_b[0] * r, mic_b[1])
    lags = utils.lag_map_2d(mic_a, mic_b, d, sr, scale, c, tol=0)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    im = ax.imshow(lags, cmap="RdYlGn", extent=[-r, r, -r, r], origin="lower")
    if add_colorbar:
        plt.colorbar(im, label="Samples difference")
    ax.scatter(
        mic_a[0],
        mic_a[1],
        marker="o",
        label=labels[0],
        c="white",
        edgecolors="black",
    )
    ax.scatter(
        mic_b[0],
        mic_b[1],
        marker="o",
        label=labels[1],
        c="black",
        edgecolors="white",
    )
    circle = plt.Circle((0, 0), r, edgecolor="black", facecolor="none")
    ax.add_artist(circle)
    if add_legend:
        ax.legend()
    return ax
