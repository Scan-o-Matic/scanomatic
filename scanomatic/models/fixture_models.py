__author__ = 'martin'

from scanomatic.generics import model


class GridHistoryModel(model.Model):

    def __init__(self, project_id="", pinning=tuple(), plate=-1,
                 center_x=-1.0, center_y=-1.0, delta_x=-1.0, delta_y=-1.0):

        self.project_id = project_id
        self.pinning = pinning
        self.plate = plate
        self.center_x = center_x
        self.center_y = center_y
        self.delta_x = delta_x
        self.delta_y = delta_y

        super(GridHistoryModel, self).__init__()