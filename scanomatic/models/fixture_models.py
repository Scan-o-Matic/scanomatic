from scanomatic.generics import model


class GrayScaleModel(model.Model):

    def __init__(self, name, targets):

        self.name = name
        self.targets = targets
        super(GrayScaleModel, self).__init__()


class GrayScaleAreaModel(model.Model):

    def __init__(self, name="", values=tuple(), width=-1.0, section_length=-1.0, x1=0, x2=0, y1=0, y2=0):

        self.name = name
        self.values = values
        self.width = width
        self.section_length = section_length
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        super(GrayScaleAreaModel, self).__init__()


class FixtureModel(model.Model):

    def __init__(self, path="", grayscale=None,
                 orientation_mark_path="", orientation_marks_x=[], orientation_marks_y=[], shape=[],
                 coordinates_scale=1, plates=[], name="", scale=1.0):

        self.name = name
        self.path = path
        self.grayscale = grayscale
        """:type : GrayScaleAreaModel"""
        self.orientation_marks_x = orientation_marks_x
        self.orientation_marks_y = orientation_marks_y
        self.shape = shape
        self.coordinates_scale = coordinates_scale
        self.plates = plates
        self.orentation_mark_path = orientation_mark_path
        self.scale = scale

        super(FixtureModel, self).__init__()


class FixturePlateModel(model.Model):

    def __init__(self, index=0, x1=0, x2=0, y1=0, y2=0):

        self.index = index
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        super(FixturePlateModel, self).__init__()
