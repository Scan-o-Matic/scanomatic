#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

def freq_analysis(Y, size_spectra=np.arange(30), detail=100.0):

    Y2 = Y - Y.mean()

    arg_factor = detail / size_spectra
    specific_range=np.arange(size_spectra.size) * detail


    D = np.zeros(int(size_spectra.size * detail))
    D_tmp = D.copy()

    for i, v in enumerate(Y2):

        D[np.round(np.array((i % size_spectra) * arg_factor +
            specific_range)).astype(np.int)] += v

        D_tmp[np.round(np.array((i % size_spectra) * arg_factor +
            specific_range)).astype(np.int)] += 1

    return D / (D_tmp+1)


def plot_9(Y, wavelengths):

    fig = plt.figure()

    for i, f in enumerate(wavelengths):

        if i == 9:

            break


        ax = fig.add_subplot(3, 3, i+1)

        ax.set_title("Wave Length {0}".format(f))
        ax.plot(freq_analysis(Y, np.array(f), Y.size))
        ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

    return fig        
