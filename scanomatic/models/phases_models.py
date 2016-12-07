import scanomatic.generics.model as model


class SegmentationModel(model.Model):

    def __init__(self, dydt=None, dydt_ranks=None, dydt_signs=None, d2yd2t=None,
                 d2yd2t_signs=None, phases=None, offset=0, log2_curve=None, times=None,
                 plate=None, pos=None):

        self.log2_curve = log2_curve
        """:type : numpy.ndarray"""
        self.times = times
        """:type : numpy.ndarray"""
        self.plate = plate
        """:type : int"""
        self.pos = pos
        """:type : (int, int)"""

        self.dydt = dydt
        """:type : numpy.ndarray"""
        self.dydt_ranks = dydt_ranks
        """:type : numpy.ndarray"""
        self.dydt_signs = dydt_signs
        """:type : numpy.ndarray"""

        self.d2yd2t = d2yd2t
        """:type : numpy.ndarray"""
        self.d2yd2t_signs = d2yd2t_signs
        """:type : numpy.ndarray"""

        self.offset = offset
        """:type : int"""

        self.phases = phases
        """:type : numpy.ndarray"""

        super(SegmentationModel, self).__init__()
