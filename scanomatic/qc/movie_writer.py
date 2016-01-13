__author__ = 'martin'
"""The following code is based on the example moviewriter presented in:

http://matplotlib.org/examples/animation/moviewriter.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


class Write_Movie(object):

    def __init__(self, target, title='Animation Plotting', artist='Scan-o-Matic', comment='', fps=15, dpi=100):

        FFMpegWriter = anim.writers['ffmpeg']
        self._writer = FFMpegWriter(fps, metadata={'title': title, 'artist': artist, 'comment':comment})
        self._target = target
        self._dpi = dpi

    def __call__(self, f, *args, **kwargs):

        print("Starting animation")
        fig = None
        for a in args:
            if isinstance(a, plt.Figure):
                fig = a

        if fig is None:
            for fig_key in ('fig', 'figure', 'Fig', 'Figure'):
                if fig_key in kwargs and isinstance(kwargs[fig_key], plt.Figure):
                    fig = kwargs[fig_key]
                    break

        if fig is None:
            fig = plt.figure()

        with self._writer.saving(fig, self._target, self._dpi):

            try:
                for _ in f(*args, **kwargs):
                    self._writer.grab_frame()
            except TypeError:
                args += (fig,)
                for _ in f(*args, **kwargs):
                    self._writer.grab_frame()

        print("Animation done!")


def demo(output_name):

    @Write_Movie(output_name)
    def _plot_method(fig, length=50):

        ax = fig.gca()
        im = ax.imshow(np.random.random([3, 3]), interpolation="nearest", vmin=0, vmax=1)

        for i in range(length):

            ax.set_title(i)
            im.set_data(np.random.random([3,3]))
            yield None