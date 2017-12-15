import types
from types import GeneratorType
import copy
import warnings
import os
from numbers import Real
import cPickle
from collections import defaultdict
from ConfigParser import ConfigParser, NoSectionError

from enum import Enum

from scanomatic.generics.model import Model
import scanomatic.generics.decorators as decorators
from scanomatic.io.logger import Logger


class UnserializationError(ValueError):
    pass


def float_list_serializer(enforce=None, serialize=None):
    if enforce is not None:
        if isinstance(enforce, types.StringTypes):
            try:
                return [float(m.strip()) for m in enforce.split(",")]
            except ValueError:
                raise UnserializationError("Could not parse '{0}' as float list".format(enforce))
        elif isinstance(enforce, list):
            return [float(e) for i, e in enumerate(enforce) if e or i < len(enforce) - 1]
        else:
            return list(enforce)

    elif serialize is not None:

        if isinstance(serialize, types.StringTypes):
            return serialize
        else:
            try:
                return ", ".join((str(e) for e in serialize))
            except TypeError:
                return str(serialize)

    else:
        return None


def email_serializer(enforce=None, serialize=None):
    if enforce is not None:
        if isinstance(enforce, types.StringTypes):
            return [m.strip() for m in enforce.split(",")]
        elif isinstance(enforce, list):
            return [str(e) for e in enforce if e]
        else:
            return list(enforce)

    elif serialize is not None:

        if isinstance(serialize, types.StringTypes):
            return serialize
        elif isinstance(serialize, list):
            return ", ".join(serialize)
        else:
            return str(serialize)

    else:
        return None


def _get_coordinates_and_items_to_validate(structure, obj):

    if obj is None or obj is False and structure[0] is not bool:
        return

    is_next_to_leaf = len(structure) == 2
    iterator = obj.iteritems() if isinstance(obj, dict) else enumerate(obj)

    try:
        for pos, item in iterator:
            if is_next_to_leaf and not (item is None or item is False and structure[1] is not bool):
                yield (pos, ), item
            elif not is_next_to_leaf:
                for coord, validation_item in _get_coordinates_and_items_to_validate(structure[1:], item):
                    yield (pos,) + coord, validation_item
    except TypeError:
        pass


def _update_object_at(obj, coordinate, value):

    if obj is None or obj is False:
        warnings.warn("Can't update None using coordinate {0} and value '{1}'".format(coordinate, value))
    if len(coordinate) == 1:
        obj[coordinate[0]] = value
    else:
        _update_object_at(obj[coordinate[0]], coordinate[1:], value)


def _toggleTuple(structure, obj, locked):

    is_next_to_leaf = len(structure) == 2
    if obj is None or obj is False and structure[0] is not bool:
        return None
    elif structure[0] is tuple:

        if not locked:
            obj = list(obj)
        if not is_next_to_leaf:
            for idx, item in enumerate(obj):
                obj[idx] = _toggleTuple(structure[1:], item, locked)
        if locked:
            obj = tuple(obj)
    elif not is_next_to_leaf:
        try:
            iterator = obj.iteritems() if isinstance(obj, dict) else enumerate(obj)
            for pos, item in iterator:
                obj[pos] = _toggleTuple(structure[1:], item, locked)

        except TypeError:
            pass
    return obj


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
        expected = set(cls.default_model.keys())
        return (
            expected.issuperset(keys) and
            len(expected.intersection(keys)) > 0
        )

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
                        return factory.create(**obj)
                    except TypeError, e:
                        cls.logger.warning("Could not use {0} on key {1} to create sub-class".format(
                            factory, obj
                        ))
                        raise e

                if index < len(factories):
                    factory = factories[index]
                    index += 1
                else:
                    break

        # noinspection PyShadowingNames
        def _enforce_other(dtype, obj):
            if obj is None or obj is False and dtype is not bool:
                return None
            elif isinstance(dtype, type) and issubclass(dtype, AbstractModelFactory):
                if isinstance(dtype, dtype.MODEL):
                    return obj
                else:
                    try:
                        return dtype.create(**obj)
                    except AttributeError:
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

            if key not in settings or settings[key] is None or key not in cls.STORE_SECTION_SERIALIZERS:
                continue

            if isinstance(cls.STORE_SECTION_SERIALIZERS[key], tuple):

                ref_settings = copy.deepcopy(settings[key])
                settings[key] = _toggleTuple(cls.STORE_SECTION_SERIALIZERS[key], settings[key], False)
                dtype_leaf = cls.STORE_SECTION_SERIALIZERS[key][-1]
                for coord, item in _get_coordinates_and_items_to_validate(cls.STORE_SECTION_SERIALIZERS[key],
                                                                          ref_settings):

                    if isinstance(dtype_leaf, type) and issubclass(dtype_leaf, Model):
                        _update_object_at(settings[key], coord, _enforce_model(cls._SUB_FACTORIES[dtype_leaf], item))

                    else:
                        _update_object_at(settings[key], coord, _enforce_other(dtype_leaf, item))

                settings[key] = _toggleTuple(cls.STORE_SECTION_SERIALIZERS[key], settings[key], True)

            elif isinstance(cls.STORE_SECTION_SERIALIZERS[key], types.FunctionType):

                settings[key] = cls.STORE_SECTION_SERIALIZERS[key](enforce=settings[key])

            elif not isinstance(settings[key], cls.STORE_SECTION_SERIALIZERS[key]):

                dtype = cls.STORE_SECTION_SERIALIZERS[key]
                if isinstance(dtype, type) and issubclass(dtype, Model) and isinstance(settings[key], dict):
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
            return cls.serializer.load_serialized_object(copy.deepcopy(cls.serializer.serialize(model)))[0]

    @classmethod
    def copy_iterable_of_model(cls, models):

        gen = (cls.copy(model) for model in models)
        if isinstance(models, GeneratorType):
            return gen
        else:
            return type(models)(gen)

    @classmethod
    def to_dict(cls, model):

        model_as_dict = dict(**model)
        keys = model_as_dict.keys()
        for k in keys:

            if k not in cls.STORE_SECTION_SERIALIZERS:
                del model_as_dict[k]
            elif k in cls.STORE_SECTION_SERIALIZERS and isinstance(cls.STORE_SECTION_SERIALIZERS[k], types.FunctionType):

                model_as_dict[k] = cls.STORE_SECTION_SERIALIZERS[k](serialize=model_as_dict[k])

            elif isinstance(model_as_dict[k], Model):

                if type(model_as_dict[k]) in cls._SUB_FACTORIES:
                    model_as_dict[k] = cls._SUB_FACTORIES[type(model_as_dict[k])].to_dict(model_as_dict[k])
                else:
                    model_as_dict[k] = AbstractModelFactory.to_dict(model_as_dict[k])

            elif k in cls.STORE_SECTION_SERIALIZERS and isinstance(cls.STORE_SECTION_SERIALIZERS[k], tuple):

                dtype = cls.STORE_SECTION_SERIALIZERS[k]
                dtype_leaf = dtype[-1]
                model_as_dict[k] = _toggleTuple(dtype, model_as_dict[k], False)
                if isinstance(dtype_leaf, type) and issubclass(dtype_leaf, Model):
                    for coord, item in _get_coordinates_and_items_to_validate(dtype,
                                                                              model_as_dict[k]):

                        _update_object_at(model_as_dict[k], coord, cls._SUB_FACTORIES[dtype_leaf].to_dict(item))

                model_as_dict[k] = _toggleTuple(dtype, model_as_dict[k], True)

        return model_as_dict

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
    def get_invalid_as_text(cls, model):

        return ", ".join(["{0}: '{1}'".format(key, model[key]) for key in cls.get_invalid_names(model)])

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
    def populate_with_default_submodels(cls, obj):
        """Keys missing models/having None will get default instances of that field if possible
        :param obj: dict | scanomatic.generics.model.Model
        """

        for key in cls.STORE_SECTION_SERIALIZERS:

            if (key not in obj or obj[key] is None) and cls.STORE_SECTION_SERIALIZERS[key] in cls._SUB_FACTORIES:

                obj[key] = cls._SUB_FACTORIES[cls.STORE_SECTION_SERIALIZERS[key]].default_model

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

            return all(pinning_format is None or pinning_format is False or _is_pinning_format(pinning_format) for
                       pinning_format in pinning_formats)

        except:

            pass

        return False

    @staticmethod
    def _is_file(path):

        return isinstance(path, types.StringTypes) and os.path.isfile(path)

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

    @staticmethod
    def _is_real_number(obj):

        return isinstance(obj, Real)


def _is_pinning_format(pinning_format):
    # noinspection PyBroadException
    try:

        return all(isinstance(val, int) and val > 0
                   for val in pinning_format) and len(pinning_format) == 2

    except:

        pass

    return False


# noinspection PyUnresolvedReferences
class _SectionsLink(object):

    _CONFIGS = defaultdict(set)
    _LINKS = {}

    def __init__(self, subfactory, submodel):

        """
        :type submodel: scanomatic.generics.model.Model
        :type subfactory: () -> AbstractModelFactory
        """
        self._subfactory = subfactory
        self._section_name = subfactory.serializer.get_section_name(submodel)
        self._locked_name = False
        _SectionsLink._LINKS[submodel] = self

    @staticmethod
    def get_link(model):

        """

        :rtype : _SectionsLink
        """
        return _SectionsLink._LINKS[model]

    @staticmethod
    def clear_links(config_parser):
        """

        :type config_parser: LinkerConfigParser
        """
        for link in _SectionsLink._CONFIGS[config_parser.id]:
            for m, l in _SectionsLink._LINKS.items():
                if link is l:
                    del _SectionsLink._LINKS[m]
                    break
        del _SectionsLink._CONFIGS[config_parser.id]

    @staticmethod
    def set_link(subfactory, submodel, config_parser):

        """

        :type config_parser: LinkerConfigParser
        """
        link = _SectionsLink(subfactory, submodel)
        link.config_parser = config_parser
        return link

    @staticmethod
    def has_link(model):

        return model in _SectionsLink._LINKS

    @property
    def config_parser(self):
        """
        :return: ConfigParser.ConfigParser
        """
        try:
            return (k for k, v in _SectionsLink._CONFIGS.items() if self in v).next()
        except StopIteration:
            return None

    @config_parser.setter
    def config_parser(self, value):

        """

        :type value: LinkerConfigParser
        """
        if not isinstance(value, LinkerConfigParser):
            raise ValueError("not a LinkerConfigParser")

        self._get_section(value)
        _SectionsLink._CONFIGS[value.id].add(self)

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

        self._locked_name = True
        self._section_name = _SectionsLink.get_next_free_section(parser, self._section_name)

        return self._section_name

    @staticmethod
    def get_next_free_section(parser, section_name):
        """

        :type section_name: str
        :type parser: LinkerConfigParser
        """
        section = "{0}{1}"
        enumerator = ''
        my_section = section_name
        sections = set(s.section if hasattr(s, 'section') else s for s in _SectionsLink._CONFIGS[parser.id])
        sections = sections.union(parser.sections())
        while my_section in sections:
            my_section = section.format(section_name, " #{0}".format(enumerator) if enumerator else enumerator)
            if my_section in sections:
                if enumerator:
                    enumerator += 1
                else:
                    enumerator = 2

        return my_section

    @staticmethod
    def add_section_for_non_link(parser, section):

        """

        :type section: str
        :type parser: LinkerConfigParser
        """
        _SectionsLink._CONFIGS[parser.id].add(section)

    def retrieve_items(self, config_parser):

        return config_parser.items(self._section_name)

    def retrieve_model(self, config_parser):

        return self._subfactory.serializer.unserialize_section(config_parser, self._section_name)

    def __getstate__(self):

        return {'_section_name': self.section, '_subfactory': self._subfactory}

    def __setstate__(self, state):

        self._section_name = state['_section_name']
        self._subfactory = state['_subfactory']
        self._locked_name = True


class LinkerConfigParser(object, ConfigParser):

    def __init__(self, id, clear_links=True, *args, **kwargs):

        ConfigParser.__init__(self, *args, **kwargs)
        self.id = id
        self._clear_links = clear_links

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self._clear_links:
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

    def dump(self, model, path, overwrite=False):

        if self._has_section_head_and_is_valid(model):

            if overwrite:

                conf = LinkerConfigParser(id=path, allow_no_value=True)
                section = self.get_section_name(model)
                self.serialize_into_conf(model, conf, section)
                return SerializationHelper.save_config(conf, path)

            else:

                with SerializationHelper.get_config(path) as conf:

                    self._purge_tree(conf, model)
                    section = self.get_section_name(model)
                    self.serialize_into_conf(model, conf, section)

                    return SerializationHelper.save_config(conf, path)

        return False

    def dump_to_filehandle(self, model, filehandle, as_if_appending=False):

        if self._has_section_head_and_is_valid(model):

            section = self.get_section_name(model)
            with LinkerConfigParser(id=id(filehandle), clear_links=False, allow_no_value=True) as conf:

                if 'r' in filehandle.mode:
                    fh_pos = filehandle.tell()
                    filehandle.seek(0)
                    conf.readfp(filehandle)
                    if as_if_appending:
                        filehandle.seek(0, 2)
                    else:
                        filehandle.seek(fh_pos)

                section = _SectionsLink.get_next_free_section(conf, section)
                _SectionsLink.add_section_for_non_link(conf, section)
                self.serialize_into_conf(model, conf, section)

                if 'r' in filehandle.mode:
                    filehandle.seek(0)
                    filehandle.truncate()

                conf.write(filehandle)
            return True
        return False

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
                    # TODO: Should really use datastructure
                    for item in obj:
                        sections.append(SerializationHelper.unserialize(item, object).section)
                except (AttributeError, TypeError):
                    pass

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

    @staticmethod
    def purge_all(path):

        with SerializationHelper.get_config(None) as conf:
            return SerializationHelper.save_config(conf, path)

    def load(self, path):

        with SerializationHelper.get_config(path) as conf:

            if conf:
                return tuple(self._unserialize(conf))

        return tuple()

    def load_first(self, path):

        with SerializationHelper.get_config(path) as conf:

            if conf:
                try:
                    return self._unserialize(conf).next()
                except StopIteration:
                    self._logger.error("No model in file '{0}'".format(path))
            else:
                self._logger.error("No file named '{0}'".format(path))
        return None

    def _unserialize(self, conf):

        for section in conf.sections():

            try:

                if self._factory.all_keys_valid(conf.options(section)):
                    yield self.unserialize_section(conf, section)

            except UnserializationError:

                self._logger.error("Parsing section '{0}': {1}".format(section, conf.options(section)))
                raise

    def unserialize_section(self, conf, section):

        factory = self._factory

        try:
            if not factory.all_keys_valid(conf.options(section)):
                self._logger.warning("{1} Refused section {0} because keys {2}".format(
                    section, factory, conf.options(section)))
                return None

        except NoSectionError:
            self._logger.warning("Refused section {0} because missing in file, though claimed to be there".format(
                section))
            return None
        model = {}

        for key, dtype in factory.STORE_SECTION_SERIALIZERS.items():

            if key in conf.options(section):

                try:
                    value = conf.get(section, key)
                except ValueError:
                    self._logger.critical("Could not parse section {0}, key {1}".format(section, key))
                    value = None

                if isinstance(dtype, tuple):

                    value = SerializationHelper.unserialize_structure(value, dtype, conf)

                elif isinstance(dtype, types.FunctionType):

                    value = SerializationHelper.unserialize(value, dtype)

                elif isinstance(dtype, type) and issubclass(dtype, Model) and value is not None:

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

        return tuple(self._unserialize(MockConfigParser(serialized_object)))

    def serialize(self, model):

        if not self._has_section_head(model):
            raise ValueError("Need a section head for serialization")

        with LinkerConfigParser(id=id(model)) as conf:

            conf = self.serialize_into_conf(model, conf, self.get_section_name(model))
            return tuple((section, {k: v for k, v in conf.items(section)}) for section in conf.sections())

    def serialize_into_conf(self, model, conf, section):

        # self._logger.info("Serializing {0} into '{1}' of {2}".format(model, section, conf))

        if conf.has_section(section):
            conf.remove_section(section)

        conf.add_section(section)

        factory = self._factory
        for key, dtype in factory.STORE_SECTION_SERIALIZERS.items():

            self._serialize_item(model, key, dtype, conf, section, factory)

        return conf

    @staticmethod
    def _serialize_item(model, key, dtype, conf, section, factory):

        obj = copy.deepcopy(model[key])

        if isinstance(dtype, tuple):

            obj = _toggleTuple(dtype, obj, False)
            dtype_leaf = dtype[-1]
            for coord, item in _get_coordinates_and_items_to_validate(dtype, model[key]):
                if isinstance(dtype_leaf, type) and issubclass(dtype_leaf, Model):
                    subfactory = factory.get_sub_factory(item)
                    link = _SectionsLink.set_link(subfactory, item, conf)
                    subfactory.serializer.serialize_into_conf(item, conf, link.section)
                    _update_object_at(obj, coord, SerializationHelper.serialize(link, _SectionsLink))
                else:
                    _update_object_at(obj, coord, SerializationHelper.serialize(item, dtype_leaf))

            conf.set(section, key, SerializationHelper.serialize_structure(obj, dtype))

        elif isinstance(dtype, type) and issubclass(dtype, Model) and obj is not None:

            subfactory = factory.get_sub_factory(obj)

            conf.set(section, key, SerializationHelper.serialize(_SectionsLink.set_link(subfactory, obj, conf),
                                                                 _SectionsLink))
            subfactory.serializer.serialize_into_conf(
                obj, conf, _SectionsLink.get_link(obj).section)

        else:
            conf.set(section, key, SerializationHelper.serialize(obj, dtype))

    def get_section_name(self, model):

        if isinstance(self._factory.STORE_SECTION_HEAD, types.StringTypes):
            return self._factory.STORE_SECTION_HEAD
        elif isinstance(self._factory.STORE_SECTION_HEAD, list):
            heads = [(str(model[head]) if model[head] is not None else '') for head in self._factory.STORE_SECTION_HEAD]
            if '' in heads:
                return ''
            else:
                return ", ".join(heads)
        elif isinstance(self._factory.STORE_SECTION_HEAD, tuple):
            for key in self._factory.STORE_SECTION_HEAD:
                try:
                    if key in model:
                        model = model[key]
                    else:
                        return ''
                except TypeError:
                    return ''

            return str(model) if model is not None else ''
        else:
            return ''


class SerializationHelper(object):

    def __new__(cls, *args):

        raise Exception("This class is static, can't be instantiated")

    @staticmethod
    def serialize_structure(obj, structure):

        if obj is None:
            return None

        elif len(structure) == 1:
            return SerializationHelper.serialize(
                obj, structure[0] if not isinstance(structure[0], type) or not issubclass(structure[0], Model)
                else _SectionsLink)
        else:
            return SerializationHelper.serialize(
                (SerializationHelper.serialize_structure(item, structure[1:]) for item in obj), structure[0])

    @staticmethod
    def serialize(obj, dtype):

        if obj is None:
            return None

        elif isinstance(dtype, type) and issubclass(dtype, Enum):

            return obj.name

        elif dtype is _SectionsLink:
            return cPickle.dumps(obj)

        elif dtype in (int, float, str, bool):

            return str(obj)
        elif isinstance(dtype, types.FunctionType):
            return cPickle.dumps(dtype(serialize=obj))
        else:
            if not isinstance(obj, dtype):
                obj = dtype(obj)
            return cPickle.dumps(obj)

    @staticmethod
    def isvalidtype(o, dtype):
        return isinstance(o, dtype) or not any({type(o), dtype}.difference((list, tuple)))

    @staticmethod
    def unserialize_structure(obj, structure, conf):

        if obj is None or obj is False and structure[0] is not bool:
            return None
        elif len(structure) == 1:
            if isinstance(structure[0], type) and issubclass(structure[0], Model):
                while not isinstance(obj, _SectionsLink) and obj is not None:
                    obj = SerializationHelper.unserialize(obj, _SectionsLink)

                if obj:
                    return obj.retrieve_model(conf)
                return obj
            else:
                return SerializationHelper.unserialize(obj, structure[0])
        else:
            outer_obj = -1
            while outer_obj is not None and not SerializationHelper.isvalidtype(outer_obj, structure[0]):
                outer_obj = SerializationHelper.unserialize(obj, structure[0])
            if outer_obj is None:
                return None
            return SerializationHelper.unserialize(
                (SerializationHelper.unserialize_structure(item, structure[1:], conf)
                 for item in outer_obj), structure[0])

    @staticmethod
    def unserialize(serialized_obj, dtype):

        """

        :type serialized_obj: str | generator
        """
        if serialized_obj is None or serialized_obj is False and dtype is not bool:
            return None

        elif isinstance(dtype, type) and issubclass(dtype, Enum):
            try:
                return dtype[serialized_obj]
            except (KeyError, SyntaxError):
                return None

        elif dtype is bool:
            try:
                return bool(eval(serialized_obj))
            except (NameError, AttributeError, SyntaxError):
                return False

        elif dtype in (int, float, types.StringTypes):
            try:
                return dtype(serialized_obj)
            except (TypeError, ValueError):
                try:
                    return dtype(eval(serialized_obj))
                except (SyntaxError, NameError, AttributeError, TypeError, ValueError):
                    return None

        elif isinstance(dtype, types.FunctionType):
            try:
                return dtype(enforce=cPickle.loads(serialized_obj))
            except (cPickle.PickleError, EOFError):
                return None

        elif isinstance(serialized_obj, types.GeneratorType):
            return dtype(serialized_obj)

        elif isinstance(serialized_obj, _SectionsLink) or isinstance(serialized_obj, dtype):
            return serialized_obj

        elif SerializationHelper.isvalidtype(serialized_obj, dtype):
            return serialized_obj

        else:
            try:
                return cPickle.loads(serialized_obj)
            except (cPickle.PickleError, TypeError):
                return None

    @staticmethod
    def get_config(path):

        """

        :rtype : LinkerConfigParser
        """
        conf = LinkerConfigParser(id=path, allow_no_value=True)

        if isinstance(path, types.StringTypes):
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
