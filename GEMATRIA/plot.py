import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob, os
from matplotlib import cm


def feature_plot(h, k, file_header, dst_folder, savefile=True, showplot=False):
    from matplotlib.gridspec import GridSpec as gs
    if k % 5 == 0:
        nrows = int(k / 5)
    else:
        nrows = int(k / 5) + 1
    ncols = 5
    grids = gs(nrows, ncols, hspace=0.5)
    fig = plt.figure(figsize=(2 * ncols, nrows*1.2))

    for i in range(k):
        x = int(i / 5)
        y = i % 5
        ax = fig.add_subplot(grids[x, y])
        ax.imshow(h[i].reshape(15, 30), aspect='auto', cmap='viridis')
        ax.set_title('feature {}'.format(i + 1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if savefile:
        plt.savefig('{}{}_rank_{}.png'.format(dst_folder, file_header, k),
                    bbox_inches='tight', dpi=160, transparent=True)
    if not showplot:
        plt.close()

