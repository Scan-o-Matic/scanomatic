import scanomatic.generics.model as model

import copy
from enum import Enum


class AbstractModelFactory(object) :

    _MODEL = model.Model
    _STORE_SECTION_HEAD = tuple()
    _STORE_SECTION_SERLIALIZERS = dict()

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

    @classmethod
    def dump(cls, model, path):

        if cls._STORE_SECTION_HEAD and cls.validate(model):

            conf = AbstractModelFactory._getConfig(path)
            serializedModel = cls.serialize(model)
            section = cls.getSectionName(model)

            if conf and serializedModel and section:

                AbstractModelFactory._updateConfig(conf, section, serializedModel)
                return AbstractModelFactory._saveConfig(conf, path)

        return False

    @classmethod
    def load(cls, path):

        conf = AbstractModelFactory._getConfig(path)

        if conf:
            for section in conf.sections():
                yield cls._unserializeSection(conf, section)

    @classmethod
    def _unserializeSection(cls, section):

        keys, vals = zip(*conf.items(section))
        return cls._parseSerialization(keys, vals)

    @classmethod
    def _parseSerialization(cls, keys, vals):

        keys = map(AbstractModelFactory._str2keyPath, keys)
        dtypes = tuple(cls._STORE_SECTION_SERLIALIZERS[key] for key in keys)
        model = cls.create(
            **{key: AbstractModelFactory._unserializeSection(val, dtype)
               for key, val, dtype in zip(keys, vals, dtypes) 
               if len(key) == 1 and 
               not issubclass(dtype, AbstractModelFactory)})

        for key, dtype in zip(keys, dtypes):

            if issubclass(dtype, AbstractModelFactory) and len(key) == 1:

                #TODO: Check this so function get right params
                setattr(model, key, dtype._parseSerialization(
                    cls._filterMemberModel(keys, vals, dtypes, key)))

        return model
    
    @classmethod
    def _filterMemberModel(cls, keys, vals, dtypes, keyFilter):

        filterLength = len(keyFilter)
        for key, val, dtype in zip(keys, vals, dtypes):

            if all(filt == k for filt, k in zip(keyFilter, key)):

                yield key[filterLength:], val

    @classmethod
    def serialize(cls, model):

        serializedModel = dict()

        for keyPath, dtype in cls._deepSerializeKeysAndTypes():
            
            serializableVal = AbstractModelFactory._getValueByPath(model,
                                                                   keyPath)
            serializedModel[AbstractModelFactory._keyPath2str(keyPath)] = \
                    AbstractModelFactory._serialize(serializableVal, dtype)

        return serializedModel 


    @classmethod
    def _deepSerializeKeysAndTypes(cls):

        for keyPath, dtype in cls._STORE_SECTION_SERLIALIZERS.items():

            if issubclass(dtype, AbstractModelFactory):
                for subKeyPath, dtype in dtype._deepSerializeKeysAndTypes():
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

    @classmethod
    def getSectionName(cls, model):

        return AbstractModelFactory._getValueByPath(model,
                                                    cls._STORE_SECTION_HEAD)

    @staticmethod
    def _getValueByPath(model, valuePath):

        ret = None
        for attr in valuePath:
            ret = getattr(model, attr)
            model = ret

        return ret

    @staticmethod
    def _getConfig(path):

        conf = ConfigParser.ConfigParser(
            allow_no_value=True)
        try:
            with open(paths, 'r') as fh:
                self._scannerConfs.readfp(fh)
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
            with open(paths, 'w') as fh:
                conf.write(fh)
        except IOError:
            return False
        return True

