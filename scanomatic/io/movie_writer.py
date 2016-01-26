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

    def __init__(self, target, title='Animation Plotting', artist='Scan-o-Matic', comment='', fps=15, dpi=100,
                 fig=None):
        """Generic movie writer for plotting functions

        :param target: Where to save the movie
        :param title: Name of the movie
        :param artist: Who made the movie
        :param comment: Any comment
        :param fps: Frames per second to be used
        :type fps: int
        :param dpi: Resolution of the film in dpi
        :param fig: The figure to be used (optional), it can also be inferred from the drawing function's signature
        or automatically created by the Write_Movie instance
        :return:

        # Usage

        The Write_Movie can either be used as decorator to a function (see movie_writer.demo) instantiated
        and manually passed the drawing function.

        The instances should be called with a drawing function as its primary argument, all other arguments
        are passed along to the drawing function.

        The drawing function needs to return an iterator (see movie_writer.demo for example)
        """

        FFMpegWriter = anim.writers['ffmpeg']
        self._writer = FFMpegWriter(fps, metadata={'title': title, 'artist': artist, 'comment':comment})
        self._target = target
        self._dpi = dpi
        self._fig = fig

    def __call__(self, drawing_function, *args, **kwargs):

        print("Starting animation")
        fig = self._fig

        if fig is None:
            fig = plt.figure()

        with self._writer.saving(fig, self._target, self._dpi):

            try:
                for _ in drawing_function():
                    self._writer.grab_frame()
            except TypeError:
                for _ in drawing_function(fig):
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