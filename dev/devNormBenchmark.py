import os
import numpy as np
import itertools
from src import resource_norm as rNorm

#Source files

baseDir = "/home/martin/Data/www/2Dnorm/triplicates/"
npySuffix = "_original.npy"
xmlSuffix = ".xml"
outPath = "/home/martin/Data/www/2Dnorm/triplicates/GalVsGal"
outFormat = ".png"

phenotypeIndex = 1
phenotypeName = "GT."

files = [
    "131123_triplicate_reptest_gal1",
    "131123_triplicate_reptest_gal2"
    #'Evo_131027_reproducibility_test_CuCl2_cycle50/analysis/rep_test_CuCl2_cycle50',
    #'10kcrosses_131024_reproducibility_test_Caffeine_5mg-ml/analysis/10kcrosses_rep_test_Caffeine',
    #'10kcrosses_131024_reproducibility_test_Clotrimazole_1uM/analysis/10kcrosses_rep_test_clot',
    #'10k_crosses_131024_reproducibility_test_CuCl2_500uM/analysis/10kcrosses_131024_rep_test_CuCl2_500uM',
    #'10kcrosses_131024_reproducibility_test_Galactose_2percent/analysis/10kcrosses_131024_rep_test_galactose_2percent',
    #'10kcrosses_131024_reproducibility_test_NaAs02_3mM/analysis/10kcrosses_rep_test_NaAsO2',
    #'130916_sample_of_10kcrosses_on_Hydroxyurea/analysis/Send_to_Leo_sample_10kcrosses_HU',
    #'130924_sample_of_10k_crosses_on_Rap_1a/analysis/130924_sample_of_10k_crosses_on_Rap_1a',
    #'130924_sample_of_10k_crosses_on_Rap_3a/analysis/130924_sample_of_10k_crosses_on_Rap_3a',
    #'130924_samples_of_10k_crosses_on_Rap_1b/analysis/130924_samples_of_10k_crosses_on_Rap_1b',
    #'130924_samples_of_10k_crosses_on_Rap_2b/analysis/130924_samples_of_10k_crosses_on_Rap_2b',
    #'130924_samples_of_10k_crosses_on_Rap_3b/analysis/130924_samples_of_10k_crosses_on_Rap_3b',
]

for f in files:

    fBase = os.path.join(baseDir, f)
    fXML = fBase + xmlSuffix
    fnpy = fBase + npySuffix
    fOutBase = os.path.join(outPath, os.path.basename(fBase)) + ".{0}".format(
        phenotypeName)

    phenotypes = rNorm.DataBridge(np.load(fnpy))

    subSampler = rNorm.SubSample(phenotypes, kernels=
                                 [rNorm.DEFAULT_CONTROL_POSITION_KERNEL] *
                                 len(phenotypes))

    #Before values
    f = rNorm.plotHeatMaps(phenotypes, measure=phenotypeIndex)
    f.savefig(fOutBase + "01.Before" + outFormat)

    eCoords = rNorm.getExperimentPosistionsCoordinates(phenotypes)
    E = rNorm.getCoordinateFiltered(phenotypes, eCoords, measure=phenotypeIndex,
                                    requireCorrelated=True)
    f = rNorm.plotPairWiseCorrelation(np.log2(E))
    f.savefig(fOutBase + "02.Before.correlation" + outFormat)

    #ControlPositions
    NA = rNorm.getControlPositionsArray(phenotypes)
    f = rNorm.plotHeatMaps(NA, measure=phenotypeIndex)
    f.savefig(fOutBase + "03.ControlPositions" + outFormat)

    f = rNorm.plotControlPhenotypesStats(phenotypes, measure=phenotypeIndex)
    f.savefig(fOutBase + "04.ControlPositions.stats" + outFormat)

    for s1 in (2, 2.5, 3):
        NA = rNorm.getControlPositionsArray(phenotypes)
        rNorm.applySigmaFilter(NA, nSigma=s1)
        f = rNorm.plotHeatMaps(NA, measure=phenotypeIndex)
        f.savefig(fOutBase + "05.ControlPositionsFiltered.nSigma{0}".format(
            s1) + outFormat)

        for s2 in (0.5, 0.75, 1, 1.25, 1.5, 1.75):
            N = rNorm.getNormalisationSurfaceWithGridData(
                NA, useAccumulated=False, smoothing=s2)

            f = rNorm.plotHeatMaps(N, measure=phenotypeIndex)
            f.savefig(fOutBase + "06.NormSurface.sigma{1}.Filter.nSigma{0}".format(
                s1, s2) + outFormat)

            ND = rNorm.applyNormalisation(phenotypes, N, updateBridge=False)
            f = rNorm.plotHeatMaps(ND, measure=phenotypeIndex)
            f.savefig(fOutBase + "07.NormedValues.NormSurface.sigma{1}.Filter.nSigma{0}".format(
                s1, s2) + outFormat)

            E = rNorm.getCoordinateFiltered(ND, eCoords, measure=phenotypeIndex,
                                            requireCorrelated=True)

            f = rNorm.plotPairWiseCorrelation(E)
            f.savefig(fOutBase + "08.After.correlation.NormSurface.sigma{1}.Filter.nSigma{0}".format(
                s1, s2) + outFormat)

"""

if (len(files) == 2):

    def uniqueStrings(*args):

        def _uniqueChars(*args):

            ref = args[0]

            for a in args[1:]:
                if a != ref:
                    return False

            return True

        assert len(args) > 1, "Need at least a pair to find unique parts"

        startP = 0

        continueChecking = True
        i = itertools.izip(*args)

        while (continueChecking):
            try:
                if _uniqueChars(i.next()):
                    startP += 1
            except StopIteration:
                continueChecking = False

        return args[0][:startP], (a[startP:] for a in args)

    commonPath, uniqueNames = uniqueStrings(*files)

    fOutBase = os.path.join(outPath, "X.Pairwise.{0}.{1}.{2}".format(
        commonPath, uniqueNames[0], uniqueNames[1]))

    def _getPaths(fileP):
        baseP = os.path.join(baseDir, fileP)
        return {'xml': baseP + xmlSuffix, 'npy': baseP + npySuffix}

    fPaths = [_getPaths(p) for p in files]

    phenotypes = [rNorm.DataBridge(np.load(fPaths[i]['npy'])) for i
                  in range(len(files))]
"""
