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


class GrayScaleModel(model.Model):

    def __init__(self, name="", sections=tuple(), width=-1.0, section_length=-1.0, x1=0, x2=0, y1=0, y2=0):

        self.name = name
        self.sections = sections
        self.width = width
        self.section_length = section_length
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        super(GrayScaleModel, self).__init__()


class FixtureModel(model.Model):

    def __init__(self, path="", grayscale_values=None, grayscale_targets=None, time=0, index=0, orientation_marks_x=[],
                 orientation_marks_y=[], shape=[], coordinates_scale=1, plates=[], name=""):

        self.name = name
        self.path = path
        self.grayscale_values = grayscale_values
        self.grayscale_targets = grayscale_targets
        self.time = time
        self.index = index
        self.orientation_marks_x = orientation_marks_x
        self.orientation_marks_y = orientation_marks_y
        self.shape = shape
        self.coordinates_scale = coordinates_scale
        self.plates = plates

        super(FixtureModel, self).__init__()


class FixturePlateModel(model.Model):

    def __init__(self, index=0, x1=0, x2=0, y1=0, y2=0):

        self.index = index
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        super(FixturePlateModel, self).__init__()