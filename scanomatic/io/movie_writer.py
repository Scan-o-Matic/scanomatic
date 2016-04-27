"""The following code is based on the example moviewriter presented in:

http://matplotlib.org/examples/animation/moviewriter.html
"""

import numpy as np
import time
from  scanomatic.generics.purge_importing import ExpiringModule


class MovieWriter(object):

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

        self._title = title
        self._artist = artist
        self._comment = comment
        self._target = target
        self._dpi = dpi
        self._fps = fps
        self._fig = fig

    def __call__(self, drawing_function):

        def wrapped(*args, **kwargs):

            start_time = time.time()
            print("Starting animation")

            with ExpiringModule('matplotlib', run_code="mod.use('Agg')") as mpl:
                with ExpiringModule('matplotlib.pyplot') as plt:
                    with ExpiringModule('matplotlib.animation') as anim:

                        writers = [a for a in anim.writers.list() if not a.endswith("_file")]
                        if not writers:
                            print("No capability of making films")
                            return

                        preference = [u'ffmpeg', u'avconv']
                        Writer = None
                        for pref in preference:
                            if pref in writers:
                                Writer = anim.writers[pref]
                                break
                        if Writer is None:
                            Writer = anim.writers[writers[0]]

                        writer = Writer(self._fps,
                                        metadata={'title': self._title,
                                                  'artist': self._artist,
                                                  'comment': self._comment})

                        fig = self._fig

                        for v in args + tuple(kwargs.values()):
                            if isinstance(v, plt.Figure):
                                fig = v
                                break

                        if fig is None:
                            fig = plt.figure()

                        with writer.saving(fig, self._target, self._dpi):

                            frame = 0
                            try:
                                iterator = drawing_function(*args, **kwargs)
                            except TypeError:
                                iterator = drawing_function(fig, *args, **kwargs)

                            for _ in iterator:
                                writer.grab_frame()
                                if frame % 42 == 0 and frame > 0:
                                    print(
                                        "Frame {0} (Movie length = {1:.2f}s, Processing for {2:.2f}s".format(
                                            frame + 1, frame / float(self._fps), time.time() - start_time))
                                frame += 1

                        print("Animation done! (Process took: {0:.2f}s)".format(time.time() - start_time))

        return wrapped


@MovieWriter("demo_film.avi")
def demo_plot_method(fig, length=50):

    ax = fig.gca()
    im = ax.imshow(np.random.random([3, 3]), interpolation="nearest", vmin=0, vmax=1)

    for i in range(length):

        ax.set_title(i)
        im.set_data(np.random.random([3, 3]))
        yield None
