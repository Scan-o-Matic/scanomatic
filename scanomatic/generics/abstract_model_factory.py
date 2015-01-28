import scanomatic.generics.model as model
import scanomatic.io.logger as logger

import copy


class AbstractModelFactory(object) :

    _MODEL = model.Model
    _LOGGER = logger.Logger("Abstract Model Factory")

    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")
    
    @staticmethod
    def toDict(model):

        return {key: copy.deepcopy(value) for key, value in model}

    @classmethod
    def _verifyCorrectModel(cls, model):

        if not isinstance(model, cls._MODEL):
            raise TypeError("Wrong model for factory {0}!={1}".format(
                cls._MODEL, model))
            return False
        return True

    @classmethod
    def create(cls, **settings):

        return cls._MODEL(**settings)

    @classmethod
    def copy(cls, model):

        if cls._verifyCorrectModel(model):
            return cls.create(**AbstractModelFactory.toDict(model))

    @classmethod
    def validate(cls, model):

        if cls._verifyCorrectModel(model):
            return all(v is True for v in cls._getValidationResults(model))

        return False

    @classmethod
    def getInvalid(cls, model):

        return (v for v in set(cls._getValidationResults(model))
                if v is not True)

    @classmethod
    def _getValidationResults(cls, model):

        return (getattr(cls, attr)(model) for attr in dir(cls)                                          
                               if attr.startswith("validate_"))

    @classmethod
    def setInvalidToDefault(cls, model):

        if cls._verifyCorrectModel(model):
            cls.setDefault(model, fields=tuple(cls.getInvalid(model)))

    @classmethod
    def setDefault(cls, model, fields=None):

        if cls._verifyCorrectModel(model):

            defaultModel = cls._MODEL()

            for attr, val in defaultModel:
                if fields is None or getattr(cls.FIELD_TYPES, attr) in fields:
                    setattr(model, attr, val)

    @classmethod
    def clamp(cls):

        pass

    @classmethod
    def _clamp(cls, model, minModel, maxModel):

        if (cls._verifyCorrectModel(model) and 
                cls._verifyCorrectModel(minModel) and
                cls._verifyCorrectModel(maxModel)):

            for attr, val in model:
                minVal = getattr(minModel, attr)
                maxVal = getattr(maxModel, attr)

                if (minVal is not None and val < minVal):
                    setattr(model, attr, minVal)
                elif (maxVal is not None and val > maxVal):
                    setattr(model, attr, maxVal)
