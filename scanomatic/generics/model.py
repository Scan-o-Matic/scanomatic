from enum import Enum

class Model(object):

    _INITIALIZED = "_initialized"
    _RESERVED_WORDS = ('keys',)
    FIELD_TYPES = None


    def __init__(self):

        content = [attribute for attribute in self]
        fields, _ = zip(*content)
        if not self._hasSetFieldTypes():
            self._setFieldTypes(fields)

        if not all(key == key.lower() for key in fields):
            raise AttributeError("Model fields may only be lower case to work with serializers {0}".format(fields()))

        if any(field in self._RESERVED_WORDS for field in fields):
            raise AttributeError("Attributes {0} are reserved and can't be defined".format(self._RESERVED_WORDS))

        if any(k for k in fields if k.startswith("_")):
            raise AttributeError("Model attributes may not be hidden")

        self._setInitialized()

    def __iter__(self):

        for attr, value in self.__dict__.iteritems():

            if not attr.startswith("_"):

                yield attr, value

    def __setattr__(self, attr, value):

        if (attr == Model._INITIALIZED):

            raise AttributeError(
                "Can't directly set model to initialized state")

        elif (self._isInitialzed() and not hasattr(self, attr)):

            raise AttributeError(
                "Can't add new attributes after initialization")
        elif (attr in self._RESERVED_WORDS):
            raise AttributeError("Can't set reserved words")
        else:

            self.__dict__[attr] = value

    def __contains__(self, item):

        return  item in self.__dict__

    def __getitem__(self, item):

        return getattr(self, item)

    def __eq__(self, other):

        for key in self.keys():

            if key not in other or not(self[key] == other[key]):

                return False

        return True

    @classmethod
    def _hasSetFieldTypes(cls):

        return ("FIELD_TYPES" in cls.__dict__ and 
                cls.__dict__["FIELD_TYPES"] is not None)

    @classmethod
    def _setFieldTypes(cls, names):

        if cls._hasSetFieldTypes():
            raise AttributeError("Can't change field types")
        else:
            cls.FIELD_TYPES = Enum(cls.__name__, {n: hash(n) for n in names})

    def _setInitialized(self):

        self.__dict__[Model._INITIALIZED] = True

    def _isInitialzed(self):

        if (Model._INITIALIZED not in self.__dict__):
            self.__dict__[Model._INITIALIZED] = False

        return self.__dict__[Model._INITIALIZED]

    def keys(self):

        return (k for k in self.__dict__.keys() if not k.startswith("_") and k != "keys")