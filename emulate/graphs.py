import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import patches as mpatches
from matplotlib.ticker import (
    AutoMinorLocator,
    AutoLocator,
    MultipleLocator,
    MaxNLocator,
)
import matplotlib.patheffects as mpe


FULL_KWARGS = dict(lw=0.5, c="k", zorder=1)
BASIS_KWARGS = dict(
    # c="lightgrey",  # I think light grey == '0.8'
    c="0.785",
    # lw=0.8,
    lw=0.84,
    zorder=-0.1,
)
PRED_KWARGS = dict(
    dash_capstyle="round",
    linestyle=(0, (0.1, 2.7)),
    lw=1.9,
    zorder=1.5,
    # path_effects=[mpe.Stroke(linewidth=2.6, foreground="k"), mpe.Normal()],
    path_effects=[mpe.withStroke(linewidth=2.6, foreground="k")],
)


def setup_rc_params(presentation=False, constrained_layout=True, usetex=True):
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = "k"

    mpl.rcdefaults()  # Set to defaults

    # mpl.rc("text", usetex=True)
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["text.usetex"] = usetex
    # mpl.rcParams["text.latex.preview"] = True
    mpl.rcParams["font.family"] = "serif"

    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["axes.edgecolor"] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams["axes.labelcolor"] = black
    mpl.rcParams["axes.titlesize"] = fontsize

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["xtick.color"] = black
    mpl.rcParams["ytick.color"] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams["xtick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams["xtick.minor.size"] = 2.4  # Default 2.0
    mpl.rcParams["ytick.minor.size"] = 2.4
    mpl.rcParams["xtick.major.size"] = 3.9  # Default 3.5
    mpl.rcParams["ytick.major.size"] = 3.9

    ppi = 72  # points per inch
    # dpi = 150
    mpl.rcParams["figure.titlesize"] = fontsize
    mpl.rcParams["figure.dpi"] = 150  # To show up reasonably in notebooks
    mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams["figure.constrained_layout.wspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.hspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.h_pad"] = 3.0 / ppi  # 3 points
    mpl.rcParams["figure.constrained_layout.w_pad"] = 3.0 / ppi

    mpl.rcParams["legend.title_fontsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams[
        "legend.edgecolor"
    ] = "inherit"  # inherits from axes.edgecolor, to match
    mpl.rcParams["legend.facecolor"] = (
        1,
        1,
        1,
        0.6,
    )  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.borderaxespad"] = 0.8
    mpl.rcParams[
        "legend.framealpha"
    ] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams[
        "patch.linewidth"
    ] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams["hatch.linewidth"] = 0.5

    # bbox = 'tight' can distort the figure size when saved (that's its purpose).
    # mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.04, dpi=350, format='png')
    mpl.rc("savefig", transparent=False, bbox=None, dpi=400, format="png")
