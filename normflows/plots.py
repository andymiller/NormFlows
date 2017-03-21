import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns


def plot_isocontours(ax, func, xlim=[-3, 3], ylim=[-4, 4], numticks=501,
                     fill=False, vectorized=True, colors=None, levels=None):
    X, Y, Z = eval_fun_2d(func, xlim, ylim, numticks, vectorized)
    if fill:
        return ax.contourf(X, Y, Z, linewidths=2, colors=colors)
    else:
        return ax.contour(X, Y, Z, linewidths=2, colors=colors, levels=levels)


def plot_function(ax, func, xlim=[-3, 3], ylim=[-4, 4], numticks=501,
                  vectorized=True, isocontours=False,
                  cmap='Blues', alpha=1.):
    X, Y, Z = eval_fun_2d(func, xlim, ylim, numticks, vectorized)
    return ax.imshow(Z, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                     cmap=cmap, alpha=alpha, origin='lower')


def eval_fun_2d(func, xlim, ylim, numticks, vectorized):
    import numpy as np
    x    = np.linspace(*xlim, num=numticks)
    y    = np.linspace(*ylim, num=numticks)
    X, Y = np.meshgrid(x, y)
    pts  = np.column_stack([X.ravel(), Y.ravel()])
    if vectorized:
        Z = func(pts).reshape(X.shape)
    else:
        Z = np.array([func(xy)
                      for xy in pyprind.prog_bar(pts)]).reshape(X.shape)
    return X, Y, Z
