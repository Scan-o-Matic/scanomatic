import numpy as np
import matplotlib.pyplot as plt


def load(path, numberOfImages, padding=2):

    im = plt.imread(path)

    if im.ndim == 3:

        if im.shape[-1] == 4:

            #Set from alpha if possible
            im = im[..., -1]

        else:

            #Else take average of all channels (RGB)
            im = im.mean(axis=-1)

    smallImShape = np.array((
        (im.shape[0] - padding * (numberOfImages - 1)) / numberOfImages,
        (im.shape[1] - padding) / 2))

    D = None

    for i in range(numberOfImages):

        d1Lower = (smallImShape[0] + padding) * i
        d1Higher = d1Lower + smallImShape[0]

        current = np.array(((
            im[d1Lower: d1Higher:, : smallImShape[1]][::-1, ...],
            im[d1Lower: d1Higher:, smallImShape[1] + padding:][::-1, ...]), ))

        if D is None:
            D = current
        else:
            D = np.r_[D, current]

    return D


def compareColonyMasks(im, mask, refMask):

    ret = {}
    mask = mask.astype(np.bool)
    refMask = refMask.astype(np.bool)
    falsePosMask = np.logical_and(mask, mask != refMask)
    falseNegMask = np.logical_and(mask == 0, mask != refMask)
    ret['falsePositiveArea'] = falsePosMask.sum()
    ret['falseNegativeArea'] = falseNegMask.sum()
    ret['refArea'] = refMask.sum()
    ret['refPixelSum'] = im[refMask].sum()
    ret['falsePositivePixelSum'] = im[falsePosMask].sum()
    ret['falseNegativePixelSum'] = im[falseNegMask].sum()

    return ret


def buildArrays(comparisonsIterable):

    images = len(comparisonsIterable)
    ret = {
        'falsePositiveArea': np.zeros((images,), dtype=np.int),
        'falseNegativeArea': np.zeros((images,), dtype=np.int),
        'refArea': np.zeros((images,), dtype=np.int),
        'refPixelSum': np.zeros((images,), dtype=np.float),
        'falsePositivePixelSum': np.zeros((images,), dtype=np.float),
        'falseNegativePixelSum': np.zeros((images,), dtype=np.float)
    }

    for i in range(images):

        ret['falsePositiveArea'][i] = comparisonsIterable[i][
            'falsePositiveArea']

        ret['falseNegativeArea'][i] = comparisonsIterable[i][
            'falseNegativeArea']

        ret['refArea'][i] = comparisonsIterable[i]['refArea']

        ret['refPixelSum'][i] = comparisonsIterable[i]['refPixelSum']

        ret['falsePositivePixelSum'][i] = comparisonsIterable[i][
            'falsePositivePixelSum']

        ret['falseNegativePixelSum'][i] = comparisonsIterable[i][
            'falseNegativePixelSum']

    return ret


def compareAnalysisArrayToReferenceImage(data, path):

    refData = load(path, data.shape[0])

    allComparisons = [compareColonyMasks(data[i][0],
                                         data[i][1],
                                         refData[i][0]) for i in range(
                                             data.shape[0])]

    return buildArrays(allComparisons)
