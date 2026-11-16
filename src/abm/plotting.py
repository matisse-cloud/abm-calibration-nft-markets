"""Shared dark plotting theme."""
import matplotlib.pyplot as plt

DARK = '#0f1117'
CARD = '#1a1d27'
BLUE = '#378ADD'
TEAL = '#1D9E75'
CORAL = '#D85A30'
AMBER = '#EF9F27'
VIOLET = '#9B6BDF'
LGRAY = '#8b8fa8'
WHITE = '#e8eaf0'
GRID = '#2a2d3e'


def style_ax(ax, title=None):
    """Apply the dark theme to one axis."""
    ax.set_facecolor(CARD)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=LGRAY, labelsize=9)
    ax.xaxis.label.set_color(LGRAY)
    ax.yaxis.label.set_color(LGRAY)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=8)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle='--')


def new_fig(*args, **kwargs):
    """plt.subplots with a dark figure background."""
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(DARK)
    return fig, axes
