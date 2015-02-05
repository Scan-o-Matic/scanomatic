from scanomatic.generics.model import Model
import scanomatic.generics.decorators as decorators
from scanomatic.io.logger import Logger

import copy
from enum import Enum
from ConfigParser import ConfigParser
import cPickle


class AbstractModelFactory(object):
    _MODEL = Model
    _SUB_FACTORIES = dict()
    STORE_SECTION_HEAD = tuple()
    STORE_SECTION_SERLIALIZERS = dict()

    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")

    # noinspection PyMethodParameters
    @decorators.classproperty
    def serializer(cls):

        return Serializer(cls)

    @classmethod
    def get_sub_factory(cls, model):

        model_type = type(model)
        if model_type not in cls._SUB_FACTORIES:
            return AbstractModelFactory
        return cls._SUB_FACTORIES[model_type]

    @staticmethod
    def to_dict(model):

        return {key: copy.deepcopy(value) for key, value in model}

    @classmethod
    def _verify_correct_model(cls, model):

        if not isinstance(model, cls._MODEL):
            raise TypeError("Wrong model for factory {0}!={1}".format(
                cls._MODEL, model))

        return True

    @classmethod
    def create(cls, **settings):

        return cls._MODEL(**settings)

    @classmethod
    def copy(cls, model):

        if cls._verify_correct_model(model):
            return cls.create(**AbstractModelFactory.to_dict(model))

    @classmethod
    def validate(cls, model):

        if cls._verify_correct_model(model):
            return all(v is True for v in cls._get_validation_results(model))

        return False

    @classmethod
    def get_invalid(cls, model):

        return (v for v in set(cls._get_validation_results(model))
                if v is not True)

    @classmethod
    def _get_validation_results(cls, model):

        return (getattr(cls, attr)(model) for attr in dir(cls) if attr.startswith("_validate"))

    @classmethod
    def set_invalid_to_default(cls, model):

        if cls._verify_correct_model(model):
            cls.set_default(model, fields=tuple(cls.get_invalid(model)))

    @classmethod
    def set_default(cls, model, fields=None):

        if cls._verify_correct_model(model):

            default_model = cls._MODEL()

            for attr, val in default_model:
                if fields is None or getattr(default_model.FIELD_TYPES, attr) in fields:
                    setattr(model, attr, val)

    @classmethod
    def clamp(cls, model):

        pass

    @classmethod
    def _clamp(cls, model, min_model, max_model):

        if (cls._verify_correct_model(model) and
                cls._verify_correct_model(min_model) and
                cls._verify_correct_model(max_model)):

            for attr, val in model:
                min_val = getattr(min_model, attr)
                max_val = getattr(max_model, attr)

                if min_val is not None and val < min_val:
                    setattr(model, attr, min_val)
                elif max_val is not None and val > max_val:
                    setattr(model, attr, max_val)

    @classmethod
    def _correct_type_and_in_bounds(cls, model, attr, dtype, min_model_caller,
                                    max_model_caller):

        if not isinstance(getattr(model, attr), dtype):

            return getattr(model.FIELD_TYPES, attr)

        elif not AbstractModelFactory._in_bounds(
                model,
                min_model_caller(model, factory=cls),
                max_model_caller(model, factory=cls),
                attr):

            return getattr(model.FIELD_TYPES, attr)

        else:

            return True

    @staticmethod
    def _in_bounds(model, lower_bounds, upper_bounds, attr):

        val = getattr(model, attr)
        min_val = getattr(lower_bounds, attr)
        max_val = getattr(upper_bounds, attr)

        if min_val is not None and val < min_val:
            return False
        elif max_val is not None and val > max_val:
            return False
        else:
            return True

    @staticmethod
    def _is_pinning_formats(pinning_formats):

        # noinspection PyBroadException
        try:

            return all(_is_pinning_format(pinningFormat) for
                       pinningFormat in pinning_formats)

        except:

            pass

        return False


def _is_pinning_format(pinning_format):
    # noinspection PyBroadException
    try:

        return all(isinstance(val, int) and val > 0
                   for val in pinning_format)

    except:

        pass

    return False


@decorators.memoize
class Serializer(object):
    def __init__(self, factory):

        self._factory = factory
        self._logger = Logger(factory.__name__)

    def dump(self, model, path):

        factory = self._factory
        valid = factory.validate(model)

        if factory.STORE_SECTION_HEAD and valid:

            conf = SerializationHelper.get_config(path)
            serialized_model = self.serialize(model)
            section = self.get_section_name(model)

            if conf and serialized_model and section:
                SerializationHelper.update_config(conf, section, serialized_model)
                return SerializationHelper.save_config(conf, path)

        if not factory.STORE_SECTION_HEAD:
            self._logger.warning("Factory does not know head for sections")
        if not valid:
            self._logger.warning("Model {0} does not have valid data".format(model))

        return False

    def load(self, path):

        conf = SerializationHelper.get_config(path)

        if conf:
            for section in conf.sections():
                yield self._unserialize_section(conf, section)

    def _unserialize_section(self, conf, section):

        keys, vals = zip(*conf.items(section))
        return self._parse_serialization(keys, vals)

    def _parse_serialization(self, keys, vals):

        factory = self._factory
        key_paths = map(SerializationHelper.get_path_from_str, keys)
        dtypes = tuple(self._get_data_type(key_path) for key_path in key_paths)

        model_dict = {key: SerializationHelper.unserialize(val, dtype)
                      for key, key_path, val, dtype in zip(keys, key_paths, vals, dtypes)
                      if not self._get_is_sub_model(dtype, key_path) and not self._get_belongs_to_sub_model(key_path)}

        model = factory.create(**model_dict)

        for key, key_path, val, dtype in zip(keys, key_paths, vals, dtypes):

            if self._get_is_sub_model(dtype, key_path):
                # TODO: Check this so function get right params
                print zip(*SerializationHelper.filter_member_model(keys, vals, key_path))
                setattr(model, key, SerializationHelper.unserialize(val, dtype).serializer._parse_serialization(
                    *zip(*SerializationHelper.filter_member_model(keys, vals, key_path))))

        return model

    def _get_is_sub_model(self, dtype, key_path):

        return dtype is not None and len(key_path) == 1 and issubclass(dtype, AbstractModelFactory)

    def _get_belongs_to_sub_model(self, key_path):

        return len(key_path) > 1

    def _get_data_type(self, key):

        factory = self._factory
        if key in factory.STORE_SECTION_SERLIALIZERS:
            return factory.STORE_SECTION_SERLIALIZERS[key]
        return None

    def serialize(self, model):

        serialized_model = dict()

        for keyPath, dtype in self._deep_serialize_keys_and_types(model):
            serializable_val = SerializationHelper.get_value_by_path(model, keyPath)
            serialized_model[SerializationHelper.get_str_from_path(keyPath)] = \
                SerializationHelper.serialize(serializable_val, dtype)

        return serialized_model

    def _deep_serialize_keys_and_types(self, model):

        factory = self._factory
        for key_path, dtype in factory.STORE_SECTION_SERLIALIZERS.items():

            if issubclass(dtype, AbstractModelFactory):

                sub_model = getattr(model, SerializationHelper.get_str_from_path(key_path))
                if dtype is AbstractModelFactory:
                    dtype = factory.get_sub_factory(sub_model)

                yield key_path, dtype

                for sub_key_path, sub_dtype in dtype.serializer._deep_serialize_keys_and_types(sub_model):
                    yield key_path + sub_key_path, sub_dtype
            else:
                yield key_path, dtype

    def get_section_name(self, model):

        return str(SerializationHelper.get_value_by_path(model, self._factory.STORE_SECTION_HEAD))


class SerializationHelper(object):
    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")

    @staticmethod
    def filter_member_model(keys, vals, key_filter):

        filter_length = len(key_filter)
        for key, val in zip(keys, vals):

            if all(filt == k for filt, k in zip(key_filter, key)):
                yield key[filter_length:], val

    @staticmethod
    def get_str_from_path(keyPath):

        return ".".join(keyPath)

    @staticmethod
    def get_path_from_str(key):

        return tuple(key.split("."))

    @staticmethod
    def serialize(obj, dtype):

        if issubclass(dtype, Enum):

            return obj.name

        elif dtype in (int, float, str, bool):

            return str(obj)

        elif issubclass(dtype, AbstractModelFactory):
            return cPickle.dumps(dtype)
        else:

            return cPickle.dumps(obj)

    @staticmethod
    def unserialize(obj, dtype):

        if dtype is not None and issubclass(dtype, Enum):
            try:
                return dtype[obj]
            except:
                return None
        elif dtype is bool:
            try:
                return eval(obj)
            except:
                return None
        elif dtype in (int, float, str):
            try:
                return dtype(obj)
            except:
                try:
                    return eval(obj)
                except:
                    return None
        else:
            try:
                return cPickle.loads(obj)
            except:
                return None

    @staticmethod
    def get_value_by_path(model, valuePath):

        ret = None
        for attr in valuePath:
            ret = getattr(model, attr)
            model = ret

        return ret

    @staticmethod
    def get_config(path):

        conf = ConfigParser(
            allow_no_value=True)
        try:
            with open(path, 'r') as fh:
                conf.readfp(fh)
        except IOError:
            pass

        return conf

    @staticmethod
    def update_config(conf, section, serializedModel):

        conf.remove_section(section)
        conf.add_section(section)
        for key, val in serializedModel.items():
            conf.set(section, key, val)

    @staticmethod
    def save_config(conf, path):

        try:
            with open(path, 'w') as fh:
                conf.write(fh)
        except IOError:
            return False
        return True