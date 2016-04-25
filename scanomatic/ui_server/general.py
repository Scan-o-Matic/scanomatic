import os
import re
from itertools import chain

from scanomatic.io.app_config import Config


def safe_directory_name(name):
    return re.match("^[A-Za-z_0-9/]*$", name) is not None


def convert_url_to_path(url):
    if url is None:
        url = ""
    else:
        url = url.split("/")
    root = Config().paths.projects_root
    return os.path.abspath(os.path.join(*chain([root], url)))


def convert_path_to_url(prefix, path):
    if prefix:
        path =  "/".join(chain([prefix], os.path.relpath(path, Config().paths.projects_root).split(os.sep)))
    else:
        path = "/".join(os.path.relpath(path, Config().paths.projects_root).split(os.sep))

    if safe_directory_name(path):
        return path
    return None


def path_is_in_jail(path):

    return Config().paths.projects_root in path
