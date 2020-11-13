import OMEGA_core as om
import numpy as np
import pandas as pd
from skimage.transform import rescale, resize

from skimage.transform import rescale, resize


def pole_aware_resize(data,
                      length,
                      reorientation=True,
                      nrows=20,
                      ncols=50,
                      npole=4,
                      flatten=True,
                      smoothen=True):
    from skimage import filters
    if reorientation:
        half_l = int(data.shape[1] / 2)
        if np.average(data[:, :half_l]) < np.average(data[:, half_l:]):
            data = np.flip(data, axis=1)
    upscaled = rescale(data, (2, 2), anti_aliasing=True)
    pole_length = int(round((0.3 / length) * upscaled.shape[1]))
    pole1 = resize(upscaled[:, :pole_length], (nrows, npole), anti_aliasing=True)
    cell_body = resize(upscaled[:, pole_length:-pole_length], (nrows, ncols - 2 * npole), anti_aliasing=True)
    pole2 = resize(upscaled[:, -pole_length:], (nrows, npole), anti_aliasing=True)
    stitched = np.concatenate([pole1, cell_body, pole2], axis=1)
    if smoothen:
        stitched = filters.gaussian(stitched, sigma=1)
    stitched = stitched / stitched.mean()
    stitched[stitched < 0] = 0

    if flatten:
        return stitched.flatten(), length
    else:
        return stitched, length










