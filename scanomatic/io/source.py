from subprocess import call, PIPE, Popen
import os

from scanomatic import get_version
from .paths import Paths
from .logger import Logger
import requests
import tempfile
from StringIO import StringIO
import zipfile


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

    # TODO: Needs smart handling of pull to not lock etc.
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
    global _logger
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
    global _logger
    path = get_source_location()
    if has_source(path) and is_under_git_control(path):

        if branch is None:
            branch = get_active_branch(path)

        if is_newest_version(branch=branch):

            if git_pull():
                return install(path)

    if not is_newest_version():

        _logger.info("Downloading fresh into temp")
        path = download(branch=branch)
        return install(path)

    else:

        _logger.info("Already newest version")
        return True


def git_version(
        git_repo='https://raw.githubusercontent.com/local-minimum/scanomatic',
        branch='master',
        suffix='scanomatic/__init__.py'):
    global _logger
    uri = "/".join((git_repo, branch, suffix))
    for line in requests.get(uri).text.split("\n"):
        if line.startswith("__version__"):
            return line.split("=")[-1].strip()

    _logger.warning("Could not access any valid version information from uri {0}".format(uri))
    return ""


def _version_parser(version=get_version()):

    return tuple(int("".join(c for c in v if c in "0123456789")) for v in version.split(".")
                 if any((c in "0123456789" and c) for c in v))


def _greatest_version(v1, v2):
    global _logger
    comparable = min(len(v) for v in (v1, v2))
    for i in range(comparable):
        if v1[i] == v2[i]:
            continue
        elif v1[i] > v2[i]:
            return v1
        else:
            return v2

    if len(v1) >= len(v2):
        return v1
    elif len(v2) > len(v1):
        return v2

    _logger.warning("None of the versions is a version!")
    return None


def is_newest_version(branch='master'):
    global _logger
    current = _version_parser()
    online_version = git_version(branch=branch)
    if current == _greatest_version(current, _version_parser(online_version)):
        _logger.info("Already using most recent version {0}".format(get_version()))
        return True
    else:
        _logger.info("There's a new version on the branch {0} available.".format(branch))
        return False
