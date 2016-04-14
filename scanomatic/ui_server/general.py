import os
import re
from itertools import chain

from scanomatic.io.app_config import Config


def safe_directory_name(name):
    return re.match("^[A-Za-z_0-9]*$", name) is not None


def convert_url_to_path(url):
    root = Config().paths.projects_root
    return os.path.abspath(os.path.join(*chain([root], url)))


def convert_path_to_url(prefix, path):
    if not safe_directory_name(path):
        return None
    if prefix:
        return "/".join(chain([prefix], os.path.relpath(path, Config().paths.projects_root).split(os.sep)))
    else:
        return "/".join(os.path.relpath(path, Config().paths.projects_root).split(os.sep))
