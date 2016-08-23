import scanomatic.generics.model as model


class SegmentationModel(model.Model):

    def __init__(self, dydt=None, dydt_ranks=None, dydt_signs=None, d2yd2t=None,
                 d2yd2t_signs=None, phases=None, offset=0, curve=None, times=None):

        self.curve = curve
        self.times = times

        self.dydt = dydt
        self.dydt_ranks = dydt_ranks
        self.dydt_signs = dydt_signs

        self.d2yd2t = d2yd2t
        self.d2yd2t_signs = d2yd2t_signs

        self.offset = offset

        self.phases = phases

        super(VersionChangesModel, self).__init__()
