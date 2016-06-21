from subprocess import call, PIPE, Popen
import os
from .paths import Paths
from .logger import Logger
import requests
import tempfile
from StringIO import StringIO
import zipfile

import scanomatic

_logger = Logger("Source Checker")


def get_source_location():

    try:
        with open(Paths().source_location_file, 'r') as fh:
            return fh.read()
    except IOError:
        return None


def has_source(path=None):

    if path is None:
        path = get_source_location()

    if path:
        return os.path.isdir(path)
    else:
        return False


def _git_root_navigator(f):

    def _wrapped(path):

        directory = os.getcwd()
        os.chdir(path)
        ret = f(path)
        os.chdir(directory)
        return ret

    return _wrapped

@_git_root_navigator
def is_under_git_control(path):

    try:
        retcode = call(['git', 'rev-parse'])
    except OSError:
        retcode = -1
    return retcode == 0

@_git_root_navigator
def get_active_branch(path):

    p = Popen(['git', 'branch', '--list'], stdout=PIPE)
    o, _ = p.communicate()
    branch = "master"
    for l in o.split("\n"):
        if l.startswith("*"):
            branch = l.strip("* ")
            break

    return branch


@_git_root_navigator
def git_pull(path):

    # Needs smart handling of pull to not lock etc.
    """
    try:
        p = Popen(['git', 'pull'], stdout=PIPE, stderr=PIPE)
        p.wait(3)
        o, e = p.communicate(None, timeout=1)

        retcode = p.poll()
        if retcode != 0:
            p.kill()
    except OSError:
        return False
    return retcode == 0
    """
    return False


def download(base_uri="https://github.com/local-minimum/scanomatic/archive", branch=None, verbose=False):

    if branch is None:
        branch = 'master'

    uri = "{0}/{1}.zip".format(base_uri, branch)
    req = requests.get(uri)

    tf = tempfile.mkdtemp(prefix="SoM_source")

    zipdata = StringIO()
    zipdata.write(req.content)

    with zipfile.ZipFile(zipdata) as zf:

        for name in zf.namelist():

            zf.extract(name, tf)
            if verbose:
                _logger.info("Extracting: {0} -> {1}".format(name, os.path.join(tf, name)))

    return os.path.join(tf, os.walk(tf).next()[1][0])


def install(source_path):

    try:
        retcode = call(['python', os.path.join(source_path, "setup.py"), "install", "--user", "--default"],
                  stderr=PIPE, stdout=PIPE)
    except OSError:
        return False

    return retcode == 0


def upgrade(branch=None):
    path = get_source_location()
    if has_source(path) and is_under_git_control(path):

        if branch is None:
            branch = get_active_branch(path)

        if not scanomatic.is_newest_version(branch=branch):

            if git_pull():
                return install(path)

    if not scanomatic.is_newest_version():

        _logger.info("Downloading fresh into temp")
        path = download(branch=branch)
        return install(path)

    else:

        _logger.info("Already newest version")
        return True