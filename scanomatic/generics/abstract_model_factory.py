from scanomatic.generics.model import Model
import scanomatic.generics.decorators as decorators

import copy
from enum import Enum
from ConfigParser import ConfigParser


class AbstractModelFactory(object):

    _MODEL = Model
    STORE_SECTION_HEAD = tuple()
    STORE_SECTION_SERLIALIZERS = dict()

    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")

    # noinspection PyMethodParameters
    @decorators.classproperty
    def serializer(cls):

        return  Serializer(cls)

    @staticmethod
    def toDict(model):

        return {key: copy.deepcopy(value) for key, value in model}

    @classmethod
    def _verifyCorrectModel(cls, model):

        if not isinstance(model, cls._MODEL):
            raise TypeError("Wrong model for factory {0}!={1}".format(
                cls._MODEL, model))

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
                               if attr.startswith("_validate"))

    @classmethod
    def setInvalidToDefault(cls, model):

        if cls._verifyCorrectModel(model):
            cls.setDefault(model, fields=tuple(cls.getInvalid(model)))

    @classmethod
    def setDefault(cls, model, fields=None):

        if cls._verifyCorrectModel(model):

            defaultModel = cls._MODEL()

            for attr, val in defaultModel:
                if fields is None or getattr(defaultModel.FIELD_TYPES, attr) in fields:
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

    @classmethod
    def _correctTypeAndInBounds(cls, model, attr, dtype, minModelCaller,
            maxModelCaller):

        if not isinstance(getattr(model, attr), dtype):

            return getattr(model.FIELD_TYPES, attr)

        elif not AbstractModelFactory._inBounds(
                model,
                minModelCaller(model, factory=cls),
                maxModelCaller(model, factory=cls),
                attr):

            return getattr(model.FIELD_TYPES, attr)

        else:

            return True 

    @staticmethod
    def _inBounds(model, lowerBounds, upperBounds, attr):

        val = getattr(model, attr)
        minVal = getattr(lowerBounds, attr)
        maxVal = getattr(upperBounds, attr)

        if minVal is not None and val < minVal:
            return False
        elif maxVal is not None and val > maxVal:
            return False
        else:
            return True

    @staticmethod
    def _isPinningFormats(pinningFormats):

        try:
    
            return all(AbstractModelFactory._isPinningFormat(pinningFormat) for
                    pinningFormat in pinningFormats)

        except:

            pass

        return False
            
    @staticmethod
    def _isPinningFormat(pinningFormat):

        try:

            return all(isinstance(val, int) and val > 0 
                       for val in pinningFormat)

        except:

            pass

        return False


@decorators.memoize
class Serializer(object):

    def __init__(self, factory):

        self._factory = factory

    def dump(self, model, path):

        factory = self._factory

        if factory.STORE_SECTION_HEAD and factory.validate(model):

            conf = Serializer._getConfig(path)
            serializedModel = self.serialize(model)
            section = self.getSectionName(model)

            if conf and serializedModel and section:

                Serializer._updateConfig(conf, section, serializedModel)
                return Serializer._saveConfig(conf, path)

        return False

    def load(self, path):

        conf = Serializer._getConfig(path)

        if conf:
            for section in conf.sections():
                yield self._unserializeSection(conf, section)

    def _unserializeSection(self, conf, section):

        keys, vals = zip(*conf.items(section))
        return self._parseSerialization(keys, vals)

    def _parseSerialization(self, keys, vals):

        factory = self._factory
        keys = map(Serializer._str2keyPath, keys)
        dtypes = tuple(factory.STORE_SECTION_SERLIALIZERS[key] for key in keys)
        model = factory.create(
            **{key: self._unserializeSection(val, dtype)
               for key, val, dtype in zip(keys, vals, dtypes)
               if len(key) == 1 and
               not issubclass(dtype, AbstractModelFactory)})

        for key, dtype in zip(keys, dtypes):

            if issubclass(dtype, AbstractModelFactory) and len(key) == 1:

                #TODO: Check this so function get right params
                setattr(model, key, dtype.serializer._parseSerialization(
                    Serializer._filterMemberModel(keys, vals, dtypes, key)))

        return model

    @staticmethod
    def _filterMemberModel(keys, vals, dtypes, keyFilter):

        filterLength = len(keyFilter)
        for key, val, dtype in zip(keys, vals, dtypes):

            if all(filt == k for filt, k in zip(keyFilter, key)):

                yield key[filterLength:], val


    def serialize(self, model):

        serializedModel = dict()

        for keyPath, dtype in self._deepSerializeKeysAndTypes():

            serializableVal = Serializer._getValueByPath(model, keyPath)
            serializedModel[Serializer._keyPath2str(keyPath)] = Serializer._serialize(serializableVal, dtype)

        return serializedModel


    def _deepSerializeKeysAndTypes(self):

        factory = self._factory
        for keyPath, dtype in factory.STORE_SECTION_SERLIALIZERS.items():

            if issubclass(dtype, AbstractModelFactory):
                for subKeyPath, dtype in dtype.serializer._deepSerializeKeysAndTypes():
                    yield keyPath + subKeyPath, dtype
            else:
                yield keyPath, dtype

    @staticmethod
    def _keyPath2str(keyPath):

        return ".".join(keyPath)

    @staticmethod
    def _str2keyPath(key):

        return tuple(key.split("."))

    @staticmethod
    def _serialize(obj, dtype):
        #maybe pickle?
        if issubclass(dtype, Enum):

            return obj.name

        else:

            return str(obj)


    @staticmethod
    def _unserialize(obj, dtype):

        if issubclass(dtype, Enum):
            try:
                return  dtype[obj]
            except:
                return None
        else:
            try:
                return dtype(obj)
            except:
                try:
                    return eval(obj)
                except:
                    return None

    def getSectionName(self, model):

        return Serializer._getValueByPath(model, self._factory.STORE_SECTION_HEAD)

    @staticmethod
    def _getValueByPath(model, valuePath):

        ret = None
        for attr in valuePath:
            ret = getattr(model, attr)
            model = ret

        return ret

    @staticmethod
    def _getConfig(path):

        conf = ConfigParser(
            allow_no_value=True)
        try:
            with open(path, 'r') as fh:
                conf.readfp(fh)
        except IOError:
            return None

        return conf

    @staticmethod
    def _updateConfig(conf, section, serializedModel):

        conf.remove_section(section)
        conf.add_section(section)
        for key, val in serializedModel.items():
            conf.set(section, key, val)

    @staticmethod
    def _saveConfig(conf, path):

        try:
            with open(path, 'w') as fh:
                conf.write(fh)
        except IOError:
            return False
        return True