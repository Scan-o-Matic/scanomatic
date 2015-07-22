from scanomatic.generics.model import Model
import scanomatic.generics.decorators as decorators
from scanomatic.io.logger import Logger

import copy
import os
from enum import Enum
from ConfigParser import ConfigParser
import cPickle
from types import GeneratorType
from collections import defaultdict


class AbstractModelFactory(object):

    MODEL = Model

    _LOGGER = None
    _SUB_FACTORIES = dict()
    STORE_SECTION_HEAD = ""
    STORE_SECTION_SERIALIZERS = dict()

    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")

    class __metaclass__(type):

        @property
        def logger(cls):
            """
            :rtype: scanomatic.io.logger.Logger
            """
            if cls._LOGGER is None:
                cls._LOGGER = Logger(cls.__name__)

            return cls._LOGGER

        @property
        def serializer(cls):

            """

            :rtype : Serializer
            """
            return Serializer(cls)

        @property
        def default_model(cls):
            """
            :param cls:
            :rtype: scanomatic.genercs.model.Model
            """
            return cls.MODEL()

    @classmethod
    def get_sub_factory(cls, model):

        model_type = type(model)
        if model_type not in cls._SUB_FACTORIES:
            cls.logger.warning("Unknown subfactory for model-type {0}".format(model_type))
            return AbstractModelFactory
        return cls._SUB_FACTORIES[model_type]

    @classmethod
    def _verify_correct_model(cls, model):

        if not isinstance(model, cls.MODEL):
            raise TypeError("Wrong model for factory {1} is not a {0}".format(
                cls.MODEL, model))

        return True

    @classmethod
    def create(cls, **settings):

        """

        :rtype : scanomatic.genercs.model.Model
        """
        valid_keys = tuple(cls.default_model.keys())

        cls.drop_keys(settings, valid_keys)
        cls.enforce_serializer_type(settings, set(valid_keys).intersection(cls.STORE_SECTION_SERIALIZERS.keys()))

        return cls.MODEL(**settings)

    @classmethod
    def all_keys_valid(cls, keys):

        return set(tuple(cls.default_model.keys())).issuperset(keys)

    @classmethod
    def drop_keys(cls, settings, valid_keys):

        keys = tuple(settings.keys())
        for key in keys:
            if key not in valid_keys:
                cls.logger.warning("Removing key \"{0}\" from {1} creation, since not among {2}".format(
                    key, cls.MODEL, tuple(valid_keys)))
                del settings[key]

    @classmethod
    def enforce_serializer_type(cls, settings, keys=tuple()):
        """Especially good for enums and Models

        :param settings:
        :param keys:
        :return:
        """

        def _enforce_model(factory, obj):
            factories = tuple(f for f in cls._SUB_FACTORIES.values() if f != factory)
            index = 0
            while True:
                if factory in cls._SUB_FACTORIES.values():
                    try:
                        return factory.MODEL(**obj)
                    except TypeError:
                        cls.logger.warning("Could not use {0} on key {1} to create sub-class".format(
                            factory, obj
                        ))

                if index < len(factories):
                    factory = factories[index]
                    index += 1
                else:
                    break

        def _enforce_other(dtype, obj):
            if obj is None:
                return
            elif issubclass(dtype, AbstractModelFactory):
                if isinstance(dtype, dtype.MODEL):
                    return obj
                else:
                    try:
                        return dtype.create(**obj)
                    except (AttributeError):
                        cls.logger.error(
                            "Contents mismatch between factory {0} and model data '{1}'".format(dtype, obj))
                        return obj
            try:
                return dtype(obj)
            except (AttributeError, ValueError, TypeError):
                try:
                    return dtype[obj]
                except (AttributeError, KeyError, IndexError, TypeError):
                    cls.logger.error(
                        "Having problems enforcing '{0}' to be type '{1}' in supplied settings '{2}'.".format(
                            obj, dtype, settings))
                    return obj

        for key in keys:
            if key not in settings or settings[key] is None:
                continue
            if (isinstance(cls.STORE_SECTION_SERIALIZERS[key], tuple)):
                dtype_outer, dtype_inner = cls.STORE_SECTION_SERIALIZERS[key]
                if dtype_outer in (tuple, list, set):
                    if issubclass(dtype_inner, Model):
                        settings[key] = dtype_outer(_enforce_model(cls._SUB_FACTORIES[dtype_inner], item)
                                                    if isinstance(item, dict) else item for item in settings[key])
                    else:

                        settings[key] = dtype_outer(_enforce_other(dtype_inner, item) for item in settings[key])

            elif not isinstance(settings[key], cls.STORE_SECTION_SERIALIZERS[key]):
                dtype = cls.STORE_SECTION_SERIALIZERS[key]
                if issubclass(dtype, Model) and isinstance(settings[key], dict):
                    settings[key] = _enforce_model(cls._SUB_FACTORIES[dtype], settings[key])
                else:
                    settings[key] = _enforce_other(dtype, settings[key])


    @classmethod
    def update(cls, model, **settings):

        for parameter, value in settings.items():

            if parameter in model:

                setattr(model, parameter, value)

    @classmethod
    def copy(cls, model):

        if cls._verify_correct_model(model):
            return cls.serializer.load_serialized_object(
                copy.deepcopy(
                    cls.serializer.serialize(model)))

    @classmethod
    def copy_iterable_of_model(cls, models):

        gen = (cls.copy(model) for model in models)
        if isinstance(models, GeneratorType):
            return gen
        else:
            return type(models)(gen)


    @classmethod
    def to_dict(cls, model):

        D = dict(**model)
        for k in D:
            if isinstance(D[k], Model):
                if type(D[k]) in cls._SUB_FACTORIES:
                    D[k] = cls._SUB_FACTORIES[type(D[k])].to_dict(D[k])
                else:
                    D[k] = AbstractModelFactory.to_dict(D[k])
            elif isinstance(cls.STORE_SECTION_SERIALIZERS[k], tuple):
                dtype_outer, dtype_inner = cls.STORE_SECTION_SERIALIZERS[k]
                if dtype_outer in (tuple, list, set) and issubclass(dtype_inner, Model):
                    D[k] = dtype_outer(cls._SUB_FACTORIES[dtype_inner].to_dict(item)
                                       if isinstance(item, Model) and dtype_inner in cls._SUB_FACTORIES else item
                                       for item in D[k])
        return D

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
    def get_invalid_names(cls, model):

        return (v.name for v in cls.get_invalid(model))

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

            default_model = cls.MODEL()

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

    @classmethod
    def _is_valid_submodel(cls, model, key):

        sub_model = getattr(model, key)
        sub_model_type = type(sub_model)
        if isinstance(sub_model, Model) and sub_model_type in cls._SUB_FACTORIES:
            return cls._SUB_FACTORIES[sub_model_type].validate(sub_model)
        return False

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

            return all(pinning_format is None or _is_pinning_format(pinning_format) for
                       pinning_format in pinning_formats)

        except:

            pass

        return False

    @staticmethod
    def _is_file(path):

        return isinstance(path, str) and os.path.isfile(path)

    @staticmethod
    def _is_tuple_or_list(obj):

        return isinstance(obj, tuple) or isinstance(obj, list)


    @staticmethod
    def _is_enum_value(obj, enum):

        if obj in enum:
            return True

        try:
            enum(obj)
        except ValueError:
            pass
        else:
            return True

        try:
            enum[obj]
        except KeyError:
            return False
        else:
            return True


def _is_pinning_format(pinning_format):
    # noinspection PyBroadException
    try:

        return all(isinstance(val, int) and val > 0
                   for val in pinning_format) and len(pinning_format) == 2

    except:

        pass

    return False


class _SectionsLink(object):

    __CONFIGS = defaultdict(set)
    __LINKS = {}

    def __init__(self, subfactory, submodel):

        """
        :type submodel: scanomatic.generics.model.Model
        :type subfactory: () -> AbstractModelFactory
        """
        self._subfactory = subfactory
        self._section_name = subfactory.serializer.get_section_name(submodel)
        self._locked_name = False
        _SectionsLink.__LINKS[submodel] = self

    @staticmethod
    def get_link(model):

        """

        :rtype : _SectionsLink
        """
        return _SectionsLink.__LINKS[model]

    @staticmethod
    def clear_links(config_parser):
        for link in _SectionsLink.__CONFIGS[config_parser]:
            for m, l in _SectionsLink.__LINKS.items():
                if link is l:
                    del _SectionsLink.__LINKS[m]
                    break
        del _SectionsLink.__CONFIGS[config_parser]

    @staticmethod
    def set_link(subfactory, submodel, config_parser):

        link = _SectionsLink(subfactory, submodel)
        link.config_parser = config_parser
        return link

    @staticmethod
    def has_link(model):

        return model in _SectionsLink.__LINKS
    
    @property
    def config_parser(self):
        """
        :return: ConfigParser.ConfigParser
        """
        try:
            return (k for k, v in _SectionsLink.__CONFIGS.items() if self in v).next()
        except StopIteration:
            return None

    @config_parser.setter
    def config_parser(self, value):

        if not isinstance(value, ConfigParser):
            raise ValueError("not a ConfigParser")

        self._get_section(value)
        _SectionsLink.__CONFIGS[value].add(self)

    @property
    def section(self):
        """

        :rtype : str
        """
        if self._locked_name:
            return self._section_name

        parser = self.config_parser
        if parser is None:
            raise AttributeError("config_parser not set")

        return self._get_section(parser)


    def _get_section(self, parser):

        if self._locked_name:
            return self._section_name

        section = "{0}{1}"
        enumerator = ''
        for other in _SectionsLink.__CONFIGS[parser]:
            my_section = section.format(self._section_name, enumerator)
            if other.section == my_section:
                if enumerator:
                    enumerator += 1
                else:
                    enumerator = 2

        self._locked_name = True
        self._section_name = section.format(self._section_name, enumerator)
        return self._section_name

    def retrieve_items(self, config_parser):

        return config_parser.items(self._section_name)

    def retrieve_model(self, config_parser):

        return self._subfactory.serializer._unserialize_section(config_parser, self._section_name)

    def __getstate__(self):

        return {'_section_name': self.section, '_subfactory': self._subfactory}

    def __setstate__(self, state):

        self._section_name = state['_section_name']
        self._subfactory = state['_subfactory']
        self._locked_name = True


class LinkerConfigParser(object, ConfigParser):

    def __init__(self, *args, **kwargs):

        ConfigParser.__init__(self, *args, **kwargs)

    def __enter__(self):


        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        _SectionsLink.clear_links(self)

    def _read(self, fp, fpname):

        val = ConfigParser._read(self, fp, fpname)
        self._nonzero = True
        return val

    def __nonzero__(self):

        return self._nonzero if hasattr(self, '_nonzero') else len(self.sections())


class MockConfigParser(object):

    def __init__(self, serialized_object):

        self._so = serialized_object

    def sections(self):

        return tuple(name for name, _ in self._so)

    def options(self, section):

        return (contents.keys() for name, contents in self._so if section == name).next()

    def items(self, section):

        return (contents.items() for name, contents in self._so if section == name).next()

    def get(self, section, item):

        return (contents[item] for name, contents in self._so if section == name).next()


@decorators.memoize
class Serializer(object):
    def __init__(self, factory):

        """

        :type factory: AbstractModelFactory
        """
        self._factory = factory
        self._logger = Logger(factory.__name__)

    def dump(self, model, path):

        if self._has_section_head_and_is_valid(model):

            with SerializationHelper.get_config(path) as conf:

                self._purge_tree(conf, model)
                section = self.get_section_name(model)
                self._serialize(model, conf, section)

                return SerializationHelper.save_config(conf, path)

        return False

    def dump_to_filehandle(self, model, filehandle):

        if self._has_section_head_and_is_valid(model):

            section = self.get_section_name(model)
            conf = ConfigParser(allow_no_value=True)

            self._serialize(model, conf, section)
            conf.write(filehandle)
            return True
        return  False

    def _has_section_head(self, model):

        head = self.get_section_name(model)
        return bool(len(head))

    def _has_section_head_and_is_valid(self, model):

        factory = self._factory
        valid = factory.validate(model)

        if self._has_section_head(model) and valid:
            return True

        if not self._has_section_head(model):
            self._logger.warning("Factory does not know head for sections")

        if not valid:
            self._logger.warning("Model {0} does not have valid data".format(model))
            for invalid in factory.get_invalid_names(model):
                self._logger.error("Faulty value in model {0} for {1} as {2}".format(
                    model, invalid, model[invalid]))

        return False

    def purge(self, model, path):

        with SerializationHelper.get_config(path) as conf:

            if conf:
                self._purge_tree(conf, model)
                return SerializationHelper.save_config(conf, path)

        return False

    def _purge_tree(self, conf, model):

        def add_if_points_to_subsection():

            obj = SerializationHelper.unserialize(conf.get(section, key), object)

            try:

                sections.append(obj.section)

            except AttributeError:

                try:

                    for item in obj:
                        sections.append(SerializationHelper.unserialize(item, object).section)
                except (AttributeError, TypeError):
                    pass

        serializers = self._factory.STORE_SECTION_SERIALIZERS
        sections = [self.get_section_name(model)]
        index = 0
        while index < len(sections):
            section = sections[index]
            if not conf.has_section(section):
                index += 1
                continue

            for key in conf.options(section):
                add_if_points_to_subsection()

            conf.remove_section(section)


    def purge_all(self, path):

        with SerializationHelper.get_config(None) as conf:
            return SerializationHelper.save_config(conf, path)

    def load(self, path):

        with SerializationHelper.get_config(path) as conf:

            if conf:
                return self._unserialize(conf)

        return tuple()

    def _unserialize(self, conf):

        return tuple(self._unserialize_section(conf, section) for section in conf.sections()
                             if self._factory.all_keys_valid(conf.options(section)))

    def _unserialize_section(self, conf, section):

        factory = self._factory

        if not factory.all_keys_valid(conf.options(section)):
            self._logger.warning("{1} Refused section {0} because keys {2}".format(
                section, factory, conf.options(section)))
            return None

        model = {}

        for key, dtype in factory.STORE_SECTION_SERIALIZERS.items():

            if key in conf.options(section):

                value = conf.get(section, key)

                if isinstance(dtype, tuple):

                    if dtype[0] in (list, tuple):

                        if issubclass(dtype[1], Model):

                            value = dtype[0](
                                SerializationHelper.unserialize(item, _SectionsLink).retrieve_model(conf)
                                if item is not None else None for item
                                in SerializationHelper.unserialize(value, dtype[0]))

                        else:

                            value = dtype[0](SerializationHelper.unserialize(item, dtype[1]) if item is not None else
                                             None for item in
                                             SerializationHelper.unserialize(value, dtype[0]))

                elif issubclass(dtype, Model) and value is not None:

                    obj = SerializationHelper.unserialize(value, _SectionsLink)
                    if isinstance(obj, _SectionsLink):
                        value = obj.retrieve_model(conf)
                    else:
                        # This handles backward compatibility when models were pickled
                        value = obj

                else:

                    value = SerializationHelper.unserialize(value, dtype)

                model[key] = value

        return factory.create(**model)

    def load_serialized_object(self, serialized_object):

        return self._unserialize(MockConfigParser(serialized_object))

    def serialize(self, model):

        if not self._has_section_head(model):
            raise ValueError("Need a section head for serialization")

        with LinkerConfigParser() as conf:

            conf = self._serialize(model, conf, self.get_section_name(model))
            return ((section, {k: v for k, v in conf.items(section)}) for section in conf.sections())

    def _serialize(self, model, conf, section):

        # self._logger.info("Serializing {0} into '{1}' of {2}".format(model, section, conf))

        if conf.has_section(section):
            conf.remove_section(section)

        conf.add_section(section)

        factory = self._factory
        for key, dtype in factory.STORE_SECTION_SERIALIZERS.items():

            self._serialize_item(model, key, dtype, conf, section, factory)

        return conf

    def _serialize_item(self, model, key, dtype, conf, section, factory):

        value = model[key]

        if isinstance(dtype, tuple):

            dtype_outer, dtype_inner = dtype
            if dtype_outer in (list, tuple, set):

                if issubclass(dtype_inner, Model):
                    links = []

                    for item in value:
                        if item is not None:
                            subfactory = factory.get_sub_factory(item)
                            links.append(_SectionsLink.set_link(subfactory, item, conf))
                            subfactory.serializer._serialize(item, conf, _SectionsLink.get_link(item).section)
                        else:
                            links.append(item)

                    conf.set(section, key, SerializationHelper.serialize(
                        (SerializationHelper.serialize(link, _SectionsLink) for link in links), dtype_outer))
                else:
                    conf.set(section, key, SerializationHelper.serialize(
                        (SerializationHelper.serialize(item, dtype_inner) if item is not None else None for item in value),
                        dtype_outer))

        elif issubclass(dtype, Model) and value is not None:

            subfactory = factory.get_sub_factory(value)

            conf.set(section, key, SerializationHelper.serialize(_SectionsLink.set_link(subfactory, value, conf),
                                                                 _SectionsLink))
            subfactory.serializer._serialize(
                value, conf, _SectionsLink.get_link(value).section)

        else:
            conf.set(section, key, SerializationHelper.serialize(value, dtype))

    def get_section_name(self, model):

        if isinstance(self._factory.STORE_SECTION_HEAD, str):
            return self._factory.STORE_SECTION_HEAD
        elif isinstance(self._factory.STORE_SECTION_HEAD, list):
            heads = [str(model[head]) for head in self._factory.STORE_SECTION_HEAD]
            if '' in heads:
                return ''
            else:
                return ", ".join(heads)
        else:
            return str(model[self._factory.STORE_SECTION_HEAD[0]])


class SerializationHelper(object):
    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")

    @staticmethod
    def serialize(obj, dtype):

        if obj is None:
            return None

        elif issubclass(dtype, Enum):

            return obj.name

        elif dtype is _SectionsLink:
            return cPickle.dumps(obj)

        elif dtype in (int, float, str, bool):

            return str(obj)

        else:
            if not isinstance(obj, dtype):
                obj = dtype(obj)
            return cPickle.dumps(obj)

    @staticmethod
    def unserialize(obj, dtype):

        if obj is None:
            return None
        elif issubclass(dtype, Enum):
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
    def get_config(path):

        """

        :rtype : LinkerConfigParser
        """
        conf = LinkerConfigParser(
            allow_no_value=True)

        if isinstance(path, str):
            try:
                with open(path, 'r') as fh:
                    conf.readfp(fh)
            except IOError:
                pass

        return conf

    @staticmethod
    def save_config(conf, path):

        try:
            with open(path, 'w') as fh:
                conf.write(fh)
        except IOError:
            return False
        return True


def rename_setting(settings, old_name, new_name):

    if old_name in settings:
        if new_name not in settings:
            settings[new_name] = settings[old_name]
        del settings[old_name]


def split_and_replace(settings, key, new_key_pattern, new_key_index_names):
    if key in settings:

        for index, new_key_index_name in enumerate(new_key_index_names):
            try:
                settings[new_key_pattern.format(new_key_index_name)] = settings[key][index]
            except (IndexError, TypeError):
                pass

        del settings[key]