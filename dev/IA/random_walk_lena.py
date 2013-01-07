import scipy.misc
import numpy as np
from matplotlib import pyplot as plt

lena = scipy.misc.lena()
filt = np.zeros(lena.shape, dtype=bool)

seed = tuple(np.random.randint(200, 400, 2))
move_help = np.array([[0, 1, 2], [3, -1, 4], [5, 6, 7]])
filt[seed] = True
max_iter = 80
i = 0



while i < max_iter:

    random_walk = True

    while random_walk:

        filt[seed] = True
        move = np.where(move_help == np.random.randint(0, 8))
        seed = (move[0] - 1 + seed[0], move[1] - 1 + seed[1])
        random_walk = np.abs(lena[seed].mean() - lena[np.where(filt)].mean()) / \
            lena.max() < np.abs(np.random.normal(scale=0.35))

    #NEW SEED
    pos_list = np.where(filt)
    pos = np.random.randint(0, len(pos_list[0]))
    seed = (pos_list[0][pos], pos_list[1][pos])
    i += 1

plt.imshow(lena, cmap=plt.cm.Greys_r, vmin=0, vmax=255)
im2 = (lena*filt).astype(np.float64)
im2[np.where(im2 == 0)] = None 
plt.imshow(im2, cmap=plt.cm.Reds_r, vmin=0, vmax=255)
plt.show()
