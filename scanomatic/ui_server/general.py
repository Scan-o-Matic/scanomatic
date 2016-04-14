import re


def safe_directory_name(name):
    return re.match("^[A-Za-z_0-9]*$", name) is not None
