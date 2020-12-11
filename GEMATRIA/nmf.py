import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
import os, glob
import pickle as pk
from .plot import feature_plot


# now conduct NMF
def NMF_test(matrix,
             n_components=[20],
             rand_state=[47],
             alpha=[0.1],
             tol=[0.001],
             dst_folder=None,
             plot=True,
             save_plots=False,
             save_models=True):
    import pickle as pk
    # test whether input matrix has the right format
    if min(matrix.shape) <= 1:
        raise ValueError('For N x M matrix both N and M should be no smaller than 2.')
    if matrix.min() < 0:
        raise ValueError('Matrix should be non-negative.')
    if np.isnan(matrix).sum() > 0:
        raise ValueError('NaNs found in matrix')

    # create dst folder
    if dst_folder is not None:
        if not os.path.isdir(dst_folder):
            os.mkdir(dst_folder)
    parameter_dict = {}
    models = {}
    for n in n_components:
        for rd in rand_state:
            for a in alpha:
                for t in tol:
                    k = '{}_comp_{}_rdstate_{}_alpha_{}_tol'.format(n, rd, a, t)
                    parameter_dict[k] = [n, rd, a, t]
    for k, par in parameter_dict.items():
        n, rd, a, t = par
        model = NMF(init='nndsvdar',
                    shuffle='True',
                    n_components=n,
                    solver='cd',
                    random_state=rd,
                    alpha=a,
                    tol=t,
                    max_iter=2000)
        w = model.fit_transform(matrix)
        h = model.components_
        residual = model.reconstruction_err_
        if plot:
            feature_plot(h, n,
                         file_header=k,
                         dst_folder=dst_folder,
                         savefile=save_plots,
                         showplot=False)
        models[k] = [model, w, residual]
    if save_models:
        pk.dump(models, open('{}all_models.pk'.format(dst_folder), 'wb'))
    return models