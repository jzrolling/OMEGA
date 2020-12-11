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
                      pole_length=0.3,# unit: micron
                      flatten=True,
                      smoothen=False):

    """
    This function assumes length-independence of the polar hemisphere associated signals.
    - - + + + + + + + + - -
    - - + + + + + + + + - -
    - - + + + + + + + + - -
    - - + + + + + + + + - -
    - - + + + + + + + + - -
    - - + + + + + + + + - -



    :param data:
    :param length:
    :param reorientation:
    :param nrows:
    :param ncols:
    :param npole:
    :param flatten:
    :param smoothen:
    :return:
    """
    from skimage import filters
    if reorientation:
        half_l = int(data.shape[1] / 2)
        if np.average(data[:, :half_l]) < np.average(data[:, half_l:]):
            data = np.flip(data, axis=1)
    upscaled = rescale(data, (2, 2), anti_aliasing=True)
    pole_length = int(round((pole_length / length) * upscaled.shape[1]))
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


def length_based_stack(cellpics, lengths, n=13, pad=0, remove_extreme_perc=0.05):
    extrema = max(5, int(len(lengths) * remove_extreme_perc))
    sorted_stack = cellpics.copy()[np.argsort(lengths)][extrema:-extrema]
    lengths_sorted = np.sort(lengths)[extrema:-extrema]
    stack_average = []
    lengths_average = []

    for i in range(n):
        id1 = int(i * (len(lengths_sorted) / n))
        id2 = int((i + 1) * (len(lengths_sorted) / n)) + pad
        averaged = np.average(sorted_stack[id1:id2], axis=0)
        stack_average.append(averaged)
        lengths_average.append(np.average(lengths_sorted[id1:id2]))
    return np.array(stack_average), np.array(lengths_average)


def fractional_sampling(cellpics, lengths,
                        n_groups=10,
                        group_size=10,
                        remove_extreme_perc=0.05):
    extrema = max(5, int(len(lengths) * remove_extreme_perc))
    sorted_stack = cellpics.copy()[np.argsort(lengths)][extrema:-extrema]
    sorted_lengths = np.sort(lengths)[extrema:-extrema]
    stack_average = []
    lengths_average = []

    sample_size = len(sorted_stack)
    window_size = int((sample_size - group_size) / n_groups)
    for i in range(n_groups):
        id1 = window_size * i
        id2 = window_size * i + group_size
        local_averaged = np.average(sorted_stack[id1:id2], axis=0)
        local_averaged_normalized = local_averaged / local_averaged.mean()
        stack_average.append(local_averaged_normalized)
        lengths_average.append(sorted_lengths[id1:id2].mean())
    return np.array(stack_average), np.array(lengths_average)


def agglomerative_autothreshold(strain_data):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import linkage
    from sklearn.decomposition import PCA

    # define distance threshold
    Z = linkage(strain_data, 'ward')
    threshold = max(Z[:, 2].max() * 0.9, 3)

    # agglomerative clustering
    agg_clusterer = AgglomerativeClustering(n_clusters=None,
                                            compute_full_tree=True,
                                            distance_threshold=threshold).fit(strain_data)
    # pca to reduce the N of dimensions
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(strain_data)

    return np.hstack([agg_clusterer.labels_.reshape(-1, 1), pca_transformed]), threshold


def moving_window_average(data, window_size=5,interpolate = False):
    if len(data) <= window_size+2:
        raise ValueError("Input array does not have enough values")
    else:
        cum_sum = np.cumsum(np.insert(data, 0, 0))
        mw_average = (cum_sum[window_size:]-cum_sum[:-window_size])/float(window_size)
        if interpolate:
            return np.interp(np.linspace(0, 1, len(data)),
                             xp=np.linspace(0, 1, len(mw_average)),
                             fp=mw_average)
        else:
            return mw_average


def GMM(data, max_ncomponent=5):
    from sklearn import mixture
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    best_gmm = None
    lowest_bic = np.inf
    bic = []
    for n_components in range(1, max_ncomponent + 1):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full')
        gmm.fit(data)
        bic.append(gmm.bic(data))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    return best_gmm, best_gmm.fit_predict(data), bic


def GMM_prediction_reformat(data, gmm_prediction):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    group_mean = []
    group_idx = []
    reformatted_prediction = np.zeros(gmm_prediction.shape)
    if gmm_prediction.max() != 0:
        for i in np.unique(gmm_prediction):
            idx = np.where(gmm_prediction == i)
            mean = data[idx].mean()
            group_mean.append(mean)
            group_idx.append(idx)
    counter = 0
    for j in np.argsort(np.array(group_mean)):
        reformatted_prediction[group_idx[j]] = counter
        counter += 1
    return reformatted_prediction