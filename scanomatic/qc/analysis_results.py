__author__ = 'martin'

import glob
import os
import re

from scanomatic.io.movie_writer import *
from scanomatic.io.paths import Paths

_pattern = re.compile(r".*_([0-9]+)_[0-9]+_[0-9]+_[0-9]+\.image.npy")

def animate_blob_detection(save_target, position=(0, 0, 0), source_location=None, fig=None, fps=3, interval=None):

    if source_location is None:
        source_location = Paths().log

    if fig is None:
        fig = plt.figure()

    pattern = os.path.join(source_location, "grid_cell_*_{0}_{1}_{2}.image.npy".format(*position))
    files = np.array(glob.glob(pattern))
    image_indices = [int(_pattern.match(f).groups()[0]) for f in files]
    index_order = np.argsort(image_indices)

    titles = ["Image", "Background", "Blob", "Trash (Now)", "Trash (Previous)"]
    if len(fig.axes) != 5:
        fig.clf()
        for i in range(5):
            ax = fig.add_subplot(2, 3, i + 1)
            ax.set_title(titles[i])

    image_ax = fig.axes[0]
    ims = []
    data = np.load(files[0])
    for i, ax in enumerate(fig.axes):
        ims.append(ax.imshow(data, interpolation='nearest', vmin=0, vmax=(3000 if i == 0 else 1)))

    @Write_Movie(save_target, fps=fps, fig=fig)
    def _plotter():

        for i in index_order:

            ims[0].set_data(np.load(files[i]))
            base_name = files[i][:-10]
            image_ax.set_title("Image (t={0:.1f})".format(
                image_indices[i] if interval is None else image_indices[i] * interval))

            for i, ending in enumerate(('.background.filter.npy', '.blob.filter.npy',
                              '.blob.trash.current.npy', '.blob.trash.old.npy')):

                ims[i + 1].set_data(np.load(base_name + ending))

            yield